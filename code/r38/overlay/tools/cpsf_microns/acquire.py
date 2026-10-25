"""Stream complete observed cell meshes and separate synapse supervision.

Never reads compartment/radius for geometry, never generates missing skeletons,
never persists a full decoded mesh, and never exports credentials in evidence.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from caveclient import CAVEclient
from .common import ROOT, read, save, sha, verify


_clients = threading.local()


def cave():
    if hasattr(_clients, 'value'):
        return _clients.value
    secret = read(ROOT / 'private/cave-secret.json')
    c = CAVEclient('minnie65_public', auth_token=secret['token'])
    c.version = 1926
    _clients.value = c
    return c


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def soma_reference(sk, row):
    origin = np.array([sk['meta']['soma_pt_' + k] for k in 'xyz'], dtype=np.float64)
    if np.isfinite(origin).all():
        return origin, int(sk['root']), 'skeleton_soma_metadata'
    # A frozen, independently queried nucleus annotation repairs missing soma
    # metadata only. Skeleton vertices/edges and synapse supervision stay intact.
    recovery = read(ROOT / 'soma_origin_recovery.json')['records'][row['root_id']]
    assert recovery['nucleus_id'] == row['nucleus_id']
    origin = np.asarray(recovery['origin_nm'], dtype=np.float64)
    assert origin.shape == (3,) and np.isfinite(origin).all(), 'Invalid recovered soma position'
    vertices = np.asarray(sk['vertices'], dtype=np.float64)
    distance = np.linalg.norm(vertices - origin, axis=1)
    node = int(distance.argmin())
    assert node == recovery['nearest_node'], 'Recovered soma node changed'
    assert distance[node] / 1000 <= recovery['maximum_distance_um'], 'Nucleus too far from skeleton'
    return origin, node, 'frozen_nucleus_detection_v0_fallback'


def one(row, config):
    rid = row['root_id']
    folder = ROOT / 'raw/cells'
    folder.mkdir(parents=True, exist_ok=True)
    path, record = folder / (rid + '.npz'), folder / (rid + '.json')
    if record.exists():
        old = read(record)
        assert sha(path) == old['sha256']
        permitted = [sha(ROOT / 'protocol.json')] + read(ROOT / 'protocol.json').get('compatible_pilot_protocols', [])
        assert old['protocol_sha256'] in permitted
        return old
    assert shutil.disk_usage(ROOT).free > 12 * 2 ** 30, 'Keep 12 GiB free'
    begin = time.time()
    c = cave()
    skpath = ROOT / 'raw/skeletons' / (rid + '.bin')
    assert sha(skpath) == row['skeleton_sha256']
    sk = c.skeleton.decompressBytesToDict(skpath.read_bytes())
    origin, soma_root, origin_source = soma_reference(sk, row)
    s = (np.asarray(sk['vertices'], dtype=np.float64) - origin) / 1000
    e = np.asarray(sk['edges'], dtype=np.int64)
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(s), len(s)))
    assert connected_components(graph, directed=False, return_labels=False) == 1, 'Disconnected skeleton'
    assert len(e) == len(s) - 1, 'Skeleton is not a tree'
    assert np.isfinite(s).all(), 'Nonfinite soma-centered skeleton'
    # All geometry is retained; compartment and service radius never enter.
    synpath = folder / (rid + '.synapses.pkl')
    if synpath.exists():
        import pandas as pd
        syn = pd.read_pickle(synpath)
    else:
        syn = c.materialize.query_table(
            'synapses_pni_2', filter_equal_dict={'post_pt_root_id': int(rid)},
            desired_resolution=[1, 1, 1], split_positions=True, limit=500000)
        assert len(syn) < 500000, 'Query possibly truncated'
        assert syn.id.is_unique and syn.post_pt_root_id.eq(int(rid)).all()
        syn.to_pickle(synpath)
    assert syn.attrs['materialization_version'] == 1926
    post = syn[['post_pt_position_' + k for k in 'xyz']].to_numpy(dtype=np.float64)
    sv = syn.post_pt_supervoxel_id.to_numpy(dtype=np.uint64)
    unique_sv, inv = np.unique(sv, return_inverse=True)
    l2 = np.asarray(c.chunkedgraph.get_roots(unique_sv, stop_layer=2, timestamp=c.materialize.get_timestamp()), dtype=np.uint64)[inv]
    mapping = {int(lid): int(node) for lid, node in zip(sk['lvl2_ids'], sk['mesh_to_skel_map'])}
    mapped = np.asarray([mapping.get(int(lid), -1) for lid in l2], dtype=np.int64)
    assert np.all(mapped >= 0), 'Synapse L2 lacks geometry mapping; fail closed'
    cv = c.info.segmentation_cloudvolume(progress=False, parallel=1)
    manifest_path = ROOT / 'raw/mesh_manifests' / (rid + '.json')
    assert sha(manifest_path) == row['mesh_manifest_sha256']
    original_fetch = cv.mesh.fetch_manifest

    def cached_manifest(segid, **kwargs):
        if int(segid) == int(rid) and kwargs.get('bbox') is None:
            return read(manifest_path)
        return original_fetch(segid, **kwargs)

    cv.mesh.fetch_manifest = cached_manifest
    mesh = cv.mesh.get(int(rid), deduplicate_chunk_boundaries=False)[int(rid)]
    v = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    assert np.isfinite(v).all() and faces.min() >= 0 and faces.max() < len(v)
    mesh_hash = array_hash(v, faces)
    # Calculate areas in bounded chunks, never duplicate the entire triangle soup.
    areas = np.empty(len(faces), dtype=np.float64)
    for at in range(0, len(faces), 100000):
        tri = v[faces[at:at + 100000]].astype(np.float64) / 1000
        areas[at:at + len(tri)] = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                       tri[:, 2] - tri[:, 0]), axis=1) / 2
    good = np.isfinite(areas) & (areas > 0)
    assert good.mean() > .99, 'Too many invalid mesh faces'
    areas[~good] = 0
    area = float(areas.sum())
    rng = np.random.default_rng(int(hashlib.sha256(('42:' + rid).encode()).hexdigest()[:16], 16))
    count = config['surface_candidates']
    chosen = rng.choice(len(faces), count, replace=True, p=areas / area)
    tri = (v[faces[chosen]].astype(np.float64) - origin) / 1000
    u = np.sqrt(rng.random((count, 1)))
    w = rng.random((count, 1))
    points = ((1 - u) * tri[:, 0] + u * (1 - w) * tri[:, 1] + u * w * tri[:, 2]).astype(np.float32)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, skeleton=s.astype(np.float32), edges=e,
                            soma_root=np.int64(soma_root), source_points=points,
                            synapse_position=((post - origin) / 1000).astype(np.float32),
                            synapse_skeleton_node=mapped, synapse_id=syn.id.to_numpy(dtype=np.int64),
                            area_um2=np.float64(area))
    temp.replace(path)
    result = dict(root_id=rid, nucleus_id=row['nucleus_id'], split=row['split'],
                  status='complete', path=str(path), sha256=sha(path),
                  protocol_sha256=sha(ROOT / 'protocol.json'),
                  skeleton_nodes=len(s), skeleton_edges=len(e), mesh_vertices=len(v), mesh_faces=len(faces),
                  mesh_array_sha256=mesh_hash, positive_area_fraction=float(good.mean()),
                  surface_area_um2=area, source_points=count, input_synapses=len(syn),
                  synapses_l2_mapped=int((mapped >= 0).sum()),
                  synapses_sha256=sha(synpath), origin_nm=origin.tolist(),
                  soma_origin_source=origin_source, soma_root_node=soma_root,
                  compartment_read=False, service_radius_read=False,
                  full_decoded_mesh_persisted=False, elapsed_seconds=time.time() - begin)
    save(record, result)
    return result


def main(limit):
    protocol = verify()
    rows = read(ROOT / 'cohort.json')
    rows.sort(key=lambda r: (r['split'] != 'train', hashlib.sha256(('42:' + r['nucleus_id']).encode()).hexdigest()))
    if limit:
        rows = rows[:limit]
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=1 if limit else 2) as pool:
        futures = {pool.submit(one, row, protocol['data']): row for row in rows}
        for i, future in enumerate(as_completed(futures)):
            row = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(json.dumps({k: result[k] for k in ['root_id', 'skeleton_nodes', 'input_synapses', 'elapsed_seconds']}), flush=True)
            except Exception as exc:
                failure = dict(root_id=row['root_id'], error_type=type(exc).__name__, error=str(exc)[:500],
                               traceback=traceback.format_exc())
                failures.append(failure)
                save(ROOT / 'status/acquisition_failures' / (row['root_id'] + '.json'), failure)
                print(json.dumps(failure), flush=True)
            save(ROOT / 'status/acquisition.json', dict(state='running', attempted=i + 1,
                 total=len(rows), complete=len(completed), failures=len(failures), pid=os.getpid()))
    # No selective dropping or replacements. Failed rows stop the training gate.
    state = 'pilot_complete' if limit else 'complete'
    if failures:
        state = 'failed'
    save(ROOT / 'status/acquisition.json', dict(state=state, attempted=len(rows), total=len(rows),
         complete=len(completed), failures=len(failures), pid=os.getpid()))
    if failures:
        raise RuntimeError(f'{len(failures)} acquisition failures; original cohort preserved')
    if not limit:
        save(ROOT / 'status/acquisition_complete.json', dict(complete=True, count=len(completed), failures=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    main(parser.parse_args().limit)

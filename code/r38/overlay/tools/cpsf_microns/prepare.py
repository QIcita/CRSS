"""One shared, label-independent geometry plus physically separate targets."""
import argparse
import hashlib
import heapq
import os
import shutil
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from .common import ROOT, backends, read, save, sha, verify


def region_pieces(s, e, max_length=4.):
    from tools.cpsf_v22_shared_region.support import chains
    paths, adj = chains(s, e)
    lookup = {tuple(sorted(map(int, edge))): k for k, edge in enumerate(e)}
    edge_regions = [[] for _ in e]
    centers, lengths = [], []
    for path in paths:
        total = sum(np.linalg.norm(s[b] - s[a]) for a, b in zip(path, path[1:]))
        if total <= 1e-12:
            continue
        count = max(1, int(np.ceil(total / max_length - 1e-10)))
        step, offset, start = total / count, 0., len(centers)
        sums, ll = np.zeros((count, 3)), np.zeros(count)
        for a, b in zip(path, path[1:]):
            delta = s[b] - s[a]
            length = np.linalg.norm(delta)
            if length <= 1e-12:
                continue
            first = max(0, min(count - 1, int(np.floor(offset / step + 1e-10))))
            last = max(0, min(count - 1, int(np.floor((offset + length) / step - 1e-10))))
            edge = lookup[tuple(sorted((int(a), int(b))))]
            for k in range(first, last + 1):
                left, right = max(offset, k * step), min(offset + length, (k + 1) * step)
                if right - left <= 1e-12:
                    continue
                t0, t1 = (left - offset) / length, (right - offset) / length
                sums[k] += (s[a] + (t0 + t1) / 2 * delta) * (right - left)
                ll[k] += right - left
                if e[edge, 0] == a:
                    edge_regions[edge].append((t0, t1, start + k))
                else:
                    edge_regions[edge].append((1 - t1, 1 - t0, start + k))
            offset += length
        centers.extend(sums / ll[:, None])
        lengths.extend(ll)
    for entry in edge_regions:
        entry.sort()
    return np.asarray(centers), np.asarray(lengths), edge_regions, adj


def project_synapses(s, e, positions, mapped_nodes, edge_regions):
    incident = [[] for _ in s]
    lengths = np.linalg.norm(s[e[:, 1]] - s[e[:, 0]], axis=1)
    for k, (a, b) in enumerate(e):
        incident[a].append((b, k))
        incident[b].append((a, k))
    owners = np.full(len(positions), -1, dtype=np.int64)
    residuals = np.full(len(positions), np.inf)
    for node in np.unique(mapped_nodes):
        # L2 identity anchors the branch. Only nearby paths are considered,
        # instead of snapping to a possibly unrelated Euclidean-nearest branch.
        reached, queue, candidates = {int(node): 0.}, [(0., int(node))], set()
        while queue:
            dist, a = heapq.heappop(queue)
            if dist > reached[a]:
                continue
            for b, k in incident[a]:
                if edge_regions[k]:
                    candidates.add(k)
                dd = dist + lengths[k]
                if dd <= 20 and dd < reached.get(int(b), np.inf):
                    reached[int(b)] = dd
                    heapq.heappush(queue, (dd, int(b)))
        ids = np.flatnonzero(mapped_nodes == node)
        es = np.asarray(sorted(candidates), dtype=np.int64)
        assert len(es), 'L2 anchor has no nonzero edge'
        start = s[e[es, 0]]
        vec = s[e[es, 1]] - start
        delta = positions[ids, None, :] - start[None]
        t = np.clip((delta * vec[None]).sum(2) / np.maximum((vec ** 2).sum(1), 1e-20), 0, 1)
        dist = np.linalg.norm(delta - t[:, :, None] * vec[None], axis=2)
        best = dist.argmin(1)
        for j, index in enumerate(ids):
            edge, at = es[best[j]], t[j, best[j]]
            options = edge_regions[edge]
            selected = next((u for lo, hi, u in options if lo - 1e-9 <= at < hi - 1e-9), options[-1][2])
            owners[index] = selected
            residuals[index] = dist[j, best[j]]
    assert np.all(owners >= 0)
    return owners, residuals


def enhance_r38(g, patch_centers, points):
    from tools.cpsf_raw_point_attention.data import grouping
    g.update(grouping(g['point_owner'].numpy(), g['patches']))
    g['raw_patch_pos'] = torch.tensor(patch_centers / .5, dtype=torch.float32)
    # Identical R28 Utonia context inputs (moments on physical .5-um voxels).
    origin = points.min(0)
    grid = np.floor((points - origin) / .5 + 1e-5).astype(np.int64)
    unique, inverse = np.unique(grid, axis=0, return_inverse=True)
    w = g['point_weight'].numpy().astype(np.float64)
    mass = np.bincount(inverse, weights=w)
    pf = g['pf'].numpy().astype(np.float64)
    mu, second, coord = np.zeros((len(mass), 4)), np.zeros((len(mass), 4)), np.zeros((len(mass), 3))
    np.add.at(mu, inverse, pf * w[:, None])
    np.add.at(second, inverse, pf ** 2 * w[:, None])
    np.add.at(coord, inverse, (points - origin) * w[:, None])
    mu /= mass[:, None]
    second /= mass[:, None]
    coord /= mass[:, None]
    g.update(ucoord=torch.tensor(coord / .5, dtype=torch.float32), ugrid=torch.from_numpy(unique),
             uinverse=torch.from_numpy(inverse), ufeat=torch.tensor(np.column_stack(
                 (mu, np.maximum(second - mu ** 2, 0), np.log1p(mass))), dtype=torch.float32))
    # Preserve fine support and expand only the SPoTr computation neighborhood.
    if len(points) / g['patches'] < 10:
        for level in range(1, 25):
            h = .5 * 2 ** level
            _, owner = np.unique(np.floor(points / h).astype(np.int64), axis=0, return_inverse=True)
            if len(points) / (owner.max() + 1) >= 10:
                break
        else:
            raise RuntimeError('Dense grouping scale exhausted')
        mass = np.bincount(owner, weights=w)
        centers = np.zeros((len(mass), 3))
        np.add.at(centers, owner, points * w[:, None])
        centers /= mass[:, None]
        rel = (points - centers[owner]) / h
        dense = dict(point_owner=torch.from_numpy(owner), patch_mass=torch.tensor(mass, dtype=torch.float32),
                     patches=len(mass), **grouping(owner, len(mass)))
        g.update({'dense_' + k: v for k, v in dense.items()},
                 dense_pf=torch.tensor(np.column_stack((rel, np.linalg.norm(rel, axis=1))), dtype=torch.float32) * (h / .5),
                 dense_raw_patch_pos=torch.tensor(centers / h, dtype=torch.float32) * (h / .5), dense_size=h)
    return g


def build(row, cfg):
    rid = row['root_id']
    record = ROOT / 'data' / (rid + '.json')
    if record.exists():
        old = read(record)
        for p, h in [('input_path', 'input_sha256'), ('label_path', 'label_sha256')]:
            assert sha(old[p]) == old[h]
        return old
    rawrecord = read(ROOT / 'raw/cells' / (rid + '.json'))
    assert sha(rawrecord['path']) == rawrecord['sha256']
    assert shutil.disk_usage(ROOT).free > 12 * 2 ** 30
    begin = time.time()
    from tools.cpsf_geometry_fps.fps import sample
    from tools.cpsf_h01_topology_repair.adapter import geometry
    from tools.cpsf_flywire_v23.data import fixed_origin
    from tools.cpsf_v22_shared_region.data import regions
    from tools.cpsf_v24_physical_conditioning.data import adapt
    with np.load(rawrecord['path']) as a:
        s, e = a['skeleton'].astype(np.float64), a['edges']
        source = a['source_points'].astype(np.float64)
        # Query, membership, and all model features are built BEFORE labels are read.
        points, weights, _, fps_audit = sample(source, cfg['fps_points'])
        origin = (source.min(0) + source.max(0)) / 2
        geo = fixed_origin(geometry(s, e, points, weights), s, e, points, weights, origin)
        geo['tile_count'] = 0
        g = adapt(regions(geo, 'joint'))
        centers, length, edge_regions, _ = region_pieces(geo['s'].numpy().astype(float), geo['e'].numpy())
        assert np.allclose(centers, g['pos'][:len(centers)].numpy(), atol=1e-4)
        assert np.allclose(length, g['length'][:len(centers)].numpy(), atol=1e-4)
        query = np.flatnonzero((length >= cfg['minimum_scored_region_length_um']) &
                              (np.linalg.norm(centers + origin, axis=1) >= cfg['minimum_scored_region_soma_distance_um']))
        assert len(query) >= 10
        shared_points = (geo['c'].numpy().astype(float)[g['point_owner'].numpy()]
                         + g['pf'][:, :3].numpy().astype(float) * .5)
        g = enhance_r38(g, geo['c'].numpy().astype(float), shared_points)
        # Native SGFormer 7-channel graph features; no radii, surface features or labels.
        graph_pos = s - s.mean(0)
        node_mass = np.zeros(len(s))
        edge_length = np.linalg.norm(s[e[:, 0]] - s[e[:, 1]], axis=1)
        for side in (0, 1):
            np.add.at(node_mass, e[:, side], .5 * edge_length)
        graph_input = dict(feat=torch.tensor(np.column_stack((graph_pos / 20, geo['sf'].numpy())), dtype=torch.float32),
                           edge_index=torch.tensor(np.concatenate((e.T, e.T[::-1]), axis=1), dtype=torch.long))
        # Utonia's existing physical-grid adapter and missing RGB/normal convention.
        pp = shared_points
        grid, inv = np.unique(np.floor((pp - pp.min(0)) / .1).astype(np.int64), axis=0, return_inverse=True)
        pmass = np.bincount(inv, weights=geo['w'].numpy())
        ppos = np.zeros((len(grid), 3))
        np.add.at(ppos, inv, pp * geo['w'].numpy()[:, None])
        ppos /= pmass[:, None]
        scaled = (ppos - pp.mean(0)) / 20
        distance, neighbor = cKDTree(ppos).query(centers[query], k=min(16, len(ppos)), workers=2)
        neighbor_weight = pmass[neighbor] / np.maximum(distance, .1)
        neighbor_weight /= neighbor_weight.sum(1, keepdims=True)
        point_input = dict(coord=torch.tensor(scaled, dtype=torch.float32),
                           feat=torch.tensor(np.column_stack((scaled, np.zeros((len(grid), 6)))), dtype=torch.float32),
                           grid_coord=torch.from_numpy(grid), query_neighbors=torch.from_numpy(neighbor),
                           query_weights=torch.tensor(neighbor_weight, dtype=torch.float32))
        input_value = dict(r38=g, graph=graph_input, point=point_input,
                           query_index=torch.from_numpy(query), length=torch.tensor(length[query], dtype=torch.float32))
        input_path = ROOT / 'data' / (rid + '.pt')
        torch.save(input_value, input_path.with_suffix('.tmp'))
        input_path.with_suffix('.tmp').replace(input_path)
        input_hash_before_labels = sha(input_path)
        positions = a['synapse_position'].astype(float) - origin
        owners, residuals = project_synapses(geo['s'].numpy().astype(float), geo['e'].numpy(),
                                             positions, a['synapse_skeleton_node'], edge_regions)
        assert residuals.max(initial=0) <= cfg['synapse_projection_max_distance_um'], 'Synapse projection exceeds 20 um'
        counts = np.bincount(owners, minlength=len(centers))
        assert counts.sum() == len(positions)
        label_path = ROOT / 'data' / (rid + '.labels.npz')
        np.savez_compressed(label_path, target=counts[query], query_index=query,
                            length=length[query], all_region_counts=counts,
                            synapse_region=owners, synapse_projection_residual_um=residuals)
    assert sha(input_path) == input_hash_before_labels
    result = dict(root_id=rid, nucleus_id=row['nucleus_id'], split=row['split'],
                  input_path=str(input_path), input_sha256=input_hash_before_labels,
                  label_path=str(label_path), label_sha256=sha(label_path),
                  raw_sha256=rawrecord['sha256'], nodes=len(s), points=len(points),
                  regions=g['nregions'], query_regions=len(query), scored_synapses=int(counts[query].sum()),
                  all_input_synapses=len(positions), max_projection_distance_um=float(residuals.max(initial=0)),
                  p95_projection_distance_um=float(np.quantile(residuals, .95)) if len(residuals) else 0,
                  fps=fps_audit, dense_size=float(g.get('dense_size', .5)),
                  input_constructed_before_targets=True, elapsed_seconds=time.time() - begin)
    save(record, result)
    return result


def main(limit):
    protocol = verify()
    backends()
    torch.set_num_threads(2)
    rows = read(ROOT / 'cohort.json')
    rows.sort(key=lambda r: (r['split'] != 'train', hashlib.sha256(('42:' + r['nucleus_id']).encode()).hexdigest()))
    if limit:
        rows = rows[:limit]
    results = []
    for i, row in enumerate(rows):
        if not (ROOT / 'raw/cells' / (row['root_id'] + '.json')).exists():
            raise RuntimeError('Acquisition incomplete for ' + row['root_id'])
        result = build(row, protocol['data'])
        results.append(result)
        save(ROOT / 'status/preparation.json', dict(state='running', complete=i + 1, total=len(rows),
             root_id=row['root_id'], pid=os.getpid(), last_seconds=result['elapsed_seconds']))
        print(row['root_id'], result['query_regions'], result['scored_synapses'], result['elapsed_seconds'], flush=True)
    save(ROOT / ('pilot_manifest.json' if limit else 'manifest.json'),
         dict(complete=True, records=results, protocol_sha256=sha(ROOT / 'protocol.json')))
    save(ROOT / 'status/preparation.json', dict(state='pilot_complete' if limit else 'complete', complete=len(rows), total=len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args.limit)
    except Exception as exc:
        save(ROOT / 'status/preparation_failure.json', dict(error_type=type(exc).__name__, error=str(exc), pid=os.getpid()))
        raise

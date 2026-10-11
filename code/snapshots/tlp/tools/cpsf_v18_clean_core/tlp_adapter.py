"""Pure inverse-format conversion of the verified NREDD V2-E cache schema.

Algebra follows the existing build_tlp_geometry_v2e_cache.py,
build_tlp_geometry_v2d_cache.py:candidate_frame_coordinates and
bridge_surface_common.py:candidate_frame (audited 2026-09-22).
This restores cached selected points, NOT the complete original mesh surface.
No candidate pair is fed to the encoder; the entire cached surface union is
constructed once, before any downstream pair-query selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def frame(start, end):
    axis = (end-start).astype(np.float32)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        raise ValueError('Degenerate legacy chart cannot be inverted safely')
    axis /= length
    bases = np.eye(3, dtype=np.float32)
    reference = bases[int(np.argmin(np.abs(bases@axis)))]
    y = reference-float(np.dot(reference, axis))*axis
    y /= max(float(np.linalg.norm(y)), 1e-6)
    z = np.cross(axis, y); z /= max(float(np.linalg.norm(z)), 1e-6)
    return np.stack((axis, y, z), axis=1)


def load_tlp_v2e(edge_path, bridge_path):
    parts, records, cached_nodes = [], [], None
    observed_edges, queries = None, None
    for kind, path, version in (
            ('edge', Path(edge_path), 'TLP CPSF-Geometry V2-E segment cylindrical atlas'),
            ('bridge', Path(bridge_path), 'TLP CPSF-Geometry V2-E candidate cylindrical evidence')):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json'].item()))
            if meta.get('input_version') != version or meta.get('observed_swc_only') is not True:
                raise ValueError('Unverified cache schema')
            for key in ('ground_truth_used', 'labels_read', 'labels_stored', 'image_attribute_channels_stored',
                        'image_attribute_channels_used_by_fps', 'synthetic_surface_points_added'):
                if meta.get(key) is not False:
                    raise ValueError('Missing or unsafe cache provenance: '+key)
            nodes = z['node_xyz'].astype(np.float32)
            indices = z['edge_index' if kind == 'edge' else 'pair_index'].astype(np.int64)
            features = z[kind+'_features'].astype(np.float32)
            masks = z[kind+'_mask'].astype(bool)
        if cached_nodes is not None and not np.array_equal(nodes, cached_nodes):
            raise ValueError('Edge and bridge coordinate mismatch')
        cached_nodes = nodes
        if features.shape[:2] != masks.shape or features.shape[-1] != 11 or len(indices) != len(features):
            raise ValueError('Malformed point/mask arrays')
        if indices.ndim != 2 or indices.shape[1] != 2 or (len(indices) and (indices.min() < 0 or indices.max() >= len(nodes))):
            raise ValueError('Malformed chart endpoints')
        if not np.isfinite(features[masks]).all() or not np.isfinite(nodes).all():
            raise ValueError('Nonfinite cached geometry')
        physical = nodes*np.array([1., 1., 2.], dtype=np.float32)
        max_error = 0.
        for pair, f, mask in zip(indices, features, masks):
            if not mask.any():
                continue
            a, b = physical[pair]; length = max(float(np.linalg.norm(b-a)), 1e-6)
            half = max(length*.5+3., 3., 1.) if kind == 'edge' else max(length*.5+12., 8.)
            transform = frame(a, b).astype(np.float64)
            scale = np.array([half, 8., 8.]); center = ((a+b)*.5).astype(np.float64)
            local = f[mask, :3].astype(np.float64)
            world = (local*scale)@transform.T+center
            replay = (world-center)@transform/scale
            max_error = max(max_error, float(np.abs(replay-local).max()))
            parts.append(world)
        if max_error > 1e-4:
            raise ValueError('Inverse chart roundtrip failed')
        records.append(dict(kind=kind, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            retained_cached_points=int(masks.sum()), normalized_roundtrip_max=max_error,
                            source_swc=meta.get('source_swc'), source_swc_sha256=meta.get('source_swc_sha256')))
        if kind == 'edge': observed_edges = indices
        else: queries = indices
    return physical, observed_edges, np.concatenate(parts) if parts else np.empty((0, 3)), queries, records

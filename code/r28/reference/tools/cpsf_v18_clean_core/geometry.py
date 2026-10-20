"""One label-free geometry preparation path for every task.

CPU preparation is cacheable and is conceptually part of the common encoder,
not a dataset-specific adapter. No task/query is accepted here. Coordinates of
both modalities must already use the same frame and units. Exact duplicate
surface locations are encoded once, with an inverse map for original rows.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryConfig:
    voxel_size: float = 0.5
    point_neighbors: int = 8
    skeleton_members: int = 8
    surface_members: int = 8
    association_radius: float = 2.0
    surface_radius: float = 1.0
    segment_candidates: int = 4
    cross_node_candidates: int = 4
    # Multi-scale proposal radii are relative to association_radius.  The
    # middle scale reproduces the V15 radius, while the smaller/larger scales
    # expose fine interfaces and longer branch support before learned routing.
    association_scales: tuple = (0.5, 1.0, 2.0)

    def __post_init__(self):
        if min(self.voxel_size, self.association_radius, self.surface_radius) <= 0:
            raise ValueError('Geometric lengths must be positive')
        if min(self.point_neighbors, self.skeleton_members, self.surface_members,
               self.segment_candidates, self.cross_node_candidates) < 1:
            raise ValueError('Neighborhood capacities must be positive')
        if not self.association_scales or any(float(x) <= 0 for x in self.association_scales):
            raise ValueError('association_scales must contain positive values')


def neighbors(reference, queries, k, radius=np.inf):
    if not len(reference):
        return np.zeros((len(queries), k), np.int64), np.zeros((len(queries), k), bool)
    distance, ids = cKDTree(reference).query(queries, k=k, distance_upper_bound=radius, workers=1)
    ids = np.asarray(ids).reshape(len(queries), k)
    valid = np.asarray(distance).reshape(len(queries), k) < np.inf
    return np.where(valid, ids, 0).astype(np.int64), valid


def prepare_geometry(skeleton_xyz, observed_edges, surface_xyz, config=GeometryConfig()):
    """Prepare ONE sample, with no labels, root marks, or candidate-pair flags.

    Scale is the median nonzero observed-edge length. Edgeless samples use
    skeleton extent, then surface extent, then 1 input unit. This is a declared
    prototype choice, not a claim of physical scale invariance to resampling.
    """
    s = np.asarray(skeleton_xyz, dtype=np.float64)
    p = np.asarray(surface_xyz, dtype=np.float64)
    raw_edges = np.asarray(observed_edges)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError('skeleton_xyz must have shape [N,3]')
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('surface_xyz must have shape [P,3], including P=0')
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError('observed_edges must have shape [E,2]')
    if not np.isfinite(s).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite geometry')
    if not np.isfinite(raw_edges).all() or not np.equal(raw_edges, np.floor(raw_edges)).all():
        raise ValueError('Noninteger edge indices')
    e = raw_edges.astype(np.int64)
    if len(e) and (e.min() < 0 or e.max() >= len(s)):
        raise ValueError('Invalid edge indices')
    e = np.unique(np.sort(e, axis=1), axis=0)
    if len(e) and (e[:, 0] == e[:, 1]).any():
        raise ValueError('Self loops are not observed geometric segments')
    length = np.linalg.norm(s[e[:, 0]] - s[e[:, 1]], axis=1)
    positive = length[length > 1e-12]
    extent = np.linalg.norm(np.ptp(s, axis=0)) if len(s) else 0.
    if extent < 1e-12 and len(p):
        extent = np.linalg.norm(np.ptp(p, axis=0))
    scale = float(np.median(positive)) if len(positive) else float(extent if extent > 1e-12 else 1.)
    origin = s.mean(0) if len(s) else (np.unique(p,axis=0).mean(0) if len(p) else np.zeros(3))
    # Point-only H01 ablation: retain physical micrometers (scale=1),
    # point-derived origin, no observed-skeleton-derived scale or topology.
    # The N>0 joint path above is unchanged from V14.
    if not len(s):scale=1.
    s = (s-origin)/scale
    p, original_to_unique = np.unique((p-origin)/scale, axis=0, return_inverse=True)
    n, count = len(s), len(p)
    degree = np.bincount(e.ravel(), minlength=n).astype(float)
    mean_length = np.zeros(n)
    for side in (0, 1):
        np.add.at(mean_length, e[:, side], length/scale)
    mean_length /= np.maximum(degree, 1)
    sf = np.column_stack((s, np.log1p(degree), np.log1p(mean_length), degree == 1, degree > 2))

    # Voxelization groups points; it does not select representatives or drop
    # low-density patches. Every unique real point enters the learned stem.
    if count:
        _, owner = np.unique(np.floor(p/config.voxel_size).astype(np.int64), axis=0, return_inverse=True)
        m = int(owner.max())+1
        patch_count = np.bincount(owner, minlength=m)
        centers = np.zeros((m, 3)); np.add.at(centers, owner, p)
        centers /= patch_count[:, None]
        # Same full-sample tree and equations; bounded temporary PCA arrays.
        k = min(count, config.point_neighbors+1); tree = cKDTree(p)
        near = np.empty((count, k), np.int64); near_valid = np.ones((count, k), bool)
        pf = np.empty((count, 7), np.float32)
        for start in range(0, count, 65536):
            sl = slice(start, min(count, start+65536))
            _, idx = tree.query(p[sl], k=k, workers=1); idx = idx.reshape(-1,k); near[sl] = idx
            offset = p[idx]-p[sl, None]
            covariance = np.einsum('nki,nkj->nij', offset, offset)/k
            eigen = np.linalg.eigvalsh(covariance).clip(0)
            relative = p[sl]-centers[owner[sl]]
            pf[sl] = np.column_stack((relative, np.linalg.norm(relative, axis=1),
                eigen[:,0]/eigen.sum(1).clip(1e-12),
                (eigen[:,2]-eigen[:,1])/eigen[:,2].clip(1e-12),
                (eigen[:,1]-eigen[:,0])/eigen[:,2].clip(1e-12)))
        del tree
    else:
        m = 0; owner = np.empty(0, np.int64); centers = np.empty((0, 3))
        patch_count = np.empty(0, np.int64); pf = np.empty((0, 7))
        near = np.empty((0, 1), np.int64); near_valid = np.empty((0, 1), bool)

    # Separate observed topology, approximate surface adjacency, and cross-modal
    # incidence. No query, label, component or task enters this construction.
    receivers, senders = [e[:, 0], e[:, 1]], [e[:, 1], e[:, 0]]
    cross = []
    segment_cross = []
    node_tree = cKDTree(s) if n else None
    segment_tree = cKDTree((s[e[:, 0]]+s[e[:, 1]])/2) if len(e) else None
    segment_xyz = (s[e[:, 0]] + s[e[:, 1]]) / 2 if len(e) else np.empty((0, 3))
    # Orient each undirected segment by its endpoint coordinates rather than
    # by the integer node ids.  Node ids may be permuted between equivalent
    # inputs; an id-based orientation would flip the direction feature and
    # violate the encoder's permutation-invariance contract.
    if len(e):
        endpoint_a, endpoint_b = s[e[:, 0]], s[e[:, 1]]
        swap = np.array([tuple(a) > tuple(b) for a, b in zip(endpoint_a, endpoint_b)])
        canonical_a = endpoint_a.copy(); canonical_b = endpoint_b.copy()
        canonical_a[swap], canonical_b[swap] = endpoint_b[swap], endpoint_a[swap]
        segment_delta = canonical_b - canonical_a
    else:
        segment_delta = np.empty((0, 3))
    segment_length = np.linalg.norm(segment_delta, axis=1) if len(e) else np.empty(0)
    segment_unit = segment_delta / np.maximum(segment_length[:, None], 1e-12) if len(e) else np.empty((0, 3))
    segment_degree = degree[e[:, 0]] + degree[e[:, 1]] if len(e) else np.empty(0)
    segment_features = (np.column_stack((segment_xyz, segment_unit,
                                         segment_length, segment_degree))
                        if len(e) else np.empty((0, 8)))
    patch_tree = cKDTree(centers) if m else None
    for start in range(0, m, 65536):
        stop = min(start+65536, m); ids = np.arange(start, stop); c = centers[start:stop]
        dist, near_patch = patch_tree.query(c, k=min(m, config.surface_members+1),
                                            distance_upper_bound=config.surface_radius)
        near_patch = np.asarray(near_patch).reshape(len(c), -1)
        valid = (near_patch < m) & (near_patch != ids[:, None])
        # Directed neighbor messages. All targets retain their own local state.
        receivers.append(np.broadcast_to(n+ids[:, None], near_patch.shape)[valid])
        senders.append(n+near_patch[valid])
        if not n:continue
        # Multi-scale node proposals.  Each scale contributes a small local
        # support; the union is deduplicated per patch before routing.
        node_lists = [[] for _ in range(len(c))]
        segment_lists = [[] for _ in range(len(c))]
        for scale in getattr(config, 'association_scales', (1.0,)):
            radius = config.association_radius * float(scale)
            k_node = min(n, config.cross_node_candidates)
            _, near_node = node_tree.query(c, k=k_node, distance_upper_bound=radius)
            near_node = np.asarray(near_node).reshape(len(c), -1)
            for row, values in zip(node_lists, near_node):
                row.extend(int(v) for v in values if int(v) < n)
            if segment_tree is None:
                continue
            k_seg = min(len(e), max(config.segment_candidates * 2, 4))
            _, near_segment = segment_tree.query(c, k=k_seg)
            near_segment = np.asarray(near_segment).reshape(len(c), -1)
            endpoints = e[near_segment]
            aa = s[endpoints[:, :, 0]]; bb = s[endpoints[:, :, 1]]
            delta = bb-aa
            distance = np.linalg.norm(c[:, None]-(aa+
                (((c[:, None]-aa)*delta).sum(-1)/np.square(delta).sum(-1).clip(1e-12))
                .clip(0, 1)[..., None]*delta), axis=-1)
            for row, segs, dists in zip(segment_lists, near_segment, distance):
                # cKDTree may return equal-distance segments in index order.
                # Segment indices are not geometric identities and change when
                # skeleton nodes/edges are permuted.  Use an undirected,
                # coordinate-derived endpoint key to make the top-k proposal
                # deterministic and permutation invariant.
                def stable_key(j):
                    u, v = s[e[int(segs[j])]]
                    if tuple(u) > tuple(v):
                        u, v = v, u
                    return (float(dists[j]), *u.tolist(), *v.tolist())
                order = sorted(range(len(segs)), key=stable_key)[:config.segment_candidates]
                row.extend(int(seg) for seg, dist in zip(segs[order], dists[order])
                           if float(dist) <= radius)
        for local_id, (nodes_for_patch, segments_for_patch) in enumerate(zip(node_lists, segment_lists)):
            nodes_for_patch = sorted(set(nodes_for_patch))
            segments_for_patch = sorted(set(segments_for_patch))
            if nodes_for_patch:
                cross.append(np.column_stack((np.asarray(nodes_for_patch, np.int64),
                                              np.full(len(nodes_for_patch), ids[local_id], np.int64))))
            if segments_for_patch:
                segment_cross.append(np.column_stack((np.asarray(segments_for_patch, np.int64),
                                                      np.full(len(segments_for_patch), ids[local_id], np.int64))))
    receiver = np.concatenate(receivers); sender = np.concatenate(senders)
    order = np.lexsort((sender, receiver)); receiver, sender = receiver[order], sender[order]
    incidence = np.concatenate(cross) if cross else np.empty((0, 2), np.int64)
    segment_incidence = np.concatenate(segment_cross) if segment_cross else np.empty((0, 2), np.int64)
    if len(incidence):
        incidence = incidence[np.lexsort((incidence[:, 1], incidence[:, 0]))]
    # The segment offload path tiles by segment anchor, so its incidence must
    # have the same anchor-major ordering contract as the node incidence.
    # Without this sort, a tile can contain anchors from several segments and
    # local ids (anchor - tile_start) index the wrong segment state.
    if len(segment_incidence):
        segment_incidence = segment_incidence[np.lexsort((segment_incidence[:, 1], segment_incidence[:, 0]))]
    positions = np.concatenate((s, centers))
    floats = dict(skeleton_features=sf, point_features=pf, point_xyz=p,
                  skeleton_xyz=s, patch_xyz=centers, positions=positions,
                  segment_xyz=segment_xyz, segment_features=segment_features)
    integers = dict(observed_edges=e, point_patch=owner, point_neighbors=near,
                    local_receiver=receiver, local_sender=sender,
                    cross_anchor=incidence[:, 0], cross_surface=incidence[:, 1],
                    cross_segment=segment_incidence[:, 0], cross_segment_surface=segment_incidence[:, 1],
                    original_to_unique=original_to_unique, patch_point_count=patch_count)
    out = {k: torch.from_numpy(np.asarray(a, np.float32)) for k, a in floats.items()}
    out.update({k: torch.from_numpy(np.asarray(a, np.int64)) for k, a in integers.items()})
    out['neighbor_valid'] = torch.from_numpy(near_valid)
    out.update(n_nodes=n, n_patches=m, n_segments=len(e), n_observed_edges=len(e), geometry_config=config,
               input_point_count=len(original_to_unique), unique_point_count=count,
               origin=origin.tolist(), input_scale=scale,
               cross_uncovered_supports=m-len(np.unique(incidence[:, 1])),
               cross_empty_anchors=n-len(np.unique(incidence[:, 0])))
    return out


def prepare_surface_geometry(surface_xyz_um, config=GeometryConfig()):
    """H01 surface-only control; accepts neither skeleton nor query features."""
    return prepare_geometry(np.empty((0,3)),np.empty((0,2),dtype=np.int64),surface_xyz_um,config)


def to_device(geometry, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in geometry.items()}


def adapt_h01(raw):
    """Format only: halo/field-of-view selection belongs to the frozen sampler."""
    return raw['skeleton_xyz_um'], raw['edge_index'], raw['surface_xyz_um']


def adapt_ted_event(event):
    """The released TED cache stores point metric XYZ in the skeleton frame."""
    as_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    parts = [as_numpy(metric) for _, metric in event['patches']]
    return as_numpy(event['xyz']), as_numpy(event['edges']), np.concatenate(parts) if parts else np.empty((0, 3))


def adapt_tlp_world(skeleton_xyz, observed_edges, world_surface_parts):
    """Only verified common-frame XYZ; normalized legacy charts are NOT accepted
    implicitly. Include bridge-region geometry before common preparation.
    """
    return skeleton_xyz, observed_edges, np.concatenate(world_surface_parts) if world_surface_parts else np.empty((0, 3))

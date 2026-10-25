"""Training-only geometric count baselines; no type or synapse input features."""
from collections import deque
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .common import ROOT, read, save, sha, verify
from .metrics import aggregate, cell_metrics


def features(row):
    assert sha(row['input_path']) == row['input_sha256']
    data = torch.load(row['input_path'], map_location='cpu', weights_only=False)
    g, q = data['r38'], data['query_index'].numpy()
    s = g['s'].numpy().astype(float)
    both = data['graph']['edge_index'].numpy()
    e = both[:, :both.shape[1] // 2].T
    edge_lengths = np.linalg.norm(s[e[:, 1]] - s[e[:, 0]], axis=1)
    raw = ROOT / 'raw/cells' / (row['root_id'] + '.npz')
    with np.load(raw) as a:
        root, area = int(a['soma_root']), float(a['area_um2'])
    adjacency = coo_matrix((np.r_[edge_lengths, edge_lengths],
                           (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(s), len(s))).tocsr()
    distance = dijkstra(adjacency, indices=root)
    neighbors = [[] for _ in s]
    for a, b in e:
        neighbors[a].append(b)
        neighbors[b].append(a)
    order = np.zeros(len(s))
    visited, queue = {root}, deque([root])
    while queue:
        a = queue.popleft()
        for b in neighbors[a]:
            if b not in visited:
                order[b] = order[a] + (len(neighbors[a]) > 2)
                visited.add(b)
                queue.append(b)
    sr, sn, sw = [g[key].numpy() for key in ['sr', 'sn', 'sw']]
    den = g['graph_mass'].numpy()
    path = np.bincount(sr, weights=distance[sn] * sw, minlength=g['nregions']) / np.maximum(den, 1e-12)
    branch_order = np.bincount(sr, weights=order[sn] * sw, minlength=g['nregions']) / np.maximum(den, 1e-12)
    centers = g['pos'].numpy()
    valid = np.flatnonzero(g['length'].numpy() > 0)
    _, nearest = cKDTree(centers[valid]).query(g['raw_patch_pos'].numpy() * .5, workers=2)
    mass = g['patch_mass'].numpy().astype(float)
    region_area = np.bincount(valid[nearest], weights=mass / mass.sum() * area, minlength=g['nregions'])
    length = data['length'].numpy().astype(float)
    region_area = np.maximum(region_area[q], .01)
    # Area is a shared-FPS surface-mass proxy, not exact mesh-to-branch area.
    x = np.column_stack((np.log1p(region_area / length), np.log1p(path[q]),
                         np.log1p(branch_order[q]), np.log1p(length)))
    with np.load(row['label_path']) as a:
        y = a['target'].astype(float)
    return x, length, region_area, y


def main():
    verify()
    rows = read(ROOT / 'manifest.json')['records']
    values = {r['root_id']: features(r) for r in rows}
    training = [values[r['root_id']] for r in rows if r['split'] == 'train']
    x = np.concatenate([v[0] for v in training])
    length = np.concatenate([v[1] for v in training])
    area = np.concatenate([v[2] for v in training])
    y = np.concatenate([v[3] for v in training])
    weights = np.concatenate([np.full(len(v[3]), 1 / len(v[3]) / len(training)) for v in training])
    mean = (x * weights[:, None]).sum(0)
    scale = np.sqrt(((x - mean)**2 * weights[:, None]).sum(0)).clip(.01)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    rate = float(np.sum(weights * y) / np.sum(weights * length))
    area_rate = float(np.sum(weights * y) / np.sum(weights * area))
    initial = np.zeros(design.shape[1])
    initial[0] = np.log(rate)
    penalty = np.array([0.] + [1e-3] * (design.shape[1] - 1))

    def objective(beta):
        log_mu = np.log(length) + design @ beta
        mu = np.exp(log_mu)
        value = np.sum(weights * (mu - y * log_mu)) + .5 * np.sum(penalty * beta**2)
        gradient = design.T @ (weights * (mu - y)) + penalty * beta
        return value, gradient

    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   bounds=[(-12, 8)] + [(-5, 5)] * (len(initial) - 1),
                   options={'maxiter': 300, 'ftol': 1e-10})
    assert fit.success, fit.message
    result = {}
    for name in ['constant_length_rate', 'constant_area_rate', 'geometry_poisson_glm']:
        result[name] = {}
        for split in ['val', 'test']:
            scores = []
            for row in rows:
                if row['split'] != split:
                    continue
                xx, ll, aa, yy = values[row['root_id']]
                if name == 'constant_length_rate':
                    expected = ll * rate
                elif name == 'constant_area_rate':
                    expected = aa * area_rate
                else:
                    d = np.column_stack((np.ones(len(xx)), (xx - mean) / scale))
                    expected = ll * np.exp(d @ fit.x)
                scores.append(cell_metrics(yy, expected, ll))
            result[name][split] = aggregate(scores)
    save(ROOT / 'geometry_baselines.json', dict(complete=True, fitting_split='train',
         validation_tuning=False, frozen_regularization=1e-3, mean=mean.tolist(), scale=scale.tolist(),
         coefficients=fit.x.tolist(), length_rate=rate, area_rate=area_rate, results=result,
         area_definition='nearest shared surface patch to skeleton-region center, area-weight proxy',
         test_results_not_used_for_model_or_protocol_changes=True))


if __name__ == '__main__':
    main()

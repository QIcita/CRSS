"""Numerical checks for regional targets and topology-anchored projection."""
import numpy as np
from .common import ROOT, backends, read, save
from .prepare import region_pieces, project_synapses
from .geometry_baselines import features


def main():
    backends()
    # A reversed edge orientation must preserve exact segment counts/exposure.
    s = np.array([[0., 0, 0], [3., 0, 0], [8., 0, 0]])
    e = np.array([[1, 0], [1, 2]])
    centers, lengths, mapping, _ = region_pieces(s, e)
    assert np.allclose(centers, [[2., 0, 0], [6., 0, 0]])
    assert np.allclose(lengths, [4., 4.])
    pos = np.array([[.1, 0, 0], [3.9, 0, 0], [4.1, 0, 0], [8., 0, 0]])
    owner, residual = project_synapses(s, e, pos, np.array([0, 1, 1, 2]), mapping)
    assert np.array_equal(owner, [0, 0, 1, 1]) and np.allclose(residual, 0)
    # Spatially adjacent branches connected only by a long path: the L2 anchor
    # prevents jumping to the other branch, even when it is Euclidean-nearest.
    s = np.array([[0., 0, 0], [10., 0, 0], [100., 0, 0],
                  [100., 1, 0], [10., 1, 0], [0., 1, 0]])
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    centers, lengths, mapping, _ = region_pieces(s, e)
    owner, residual = project_synapses(s, e, np.array([[1., .9, 0]]), np.array([0]), mapping)
    assert centers[owner[0], 1] == 0 and np.allclose(residual, [.9])
    assert np.isclose(lengths.sum(), np.linalg.norm(s[e[:, 0]] - s[e[:, 1]], axis=1).sum())
    row = read(ROOT / 'pilot_manifest.json')['records'][0]
    assert row['split'] == 'train'
    x, ll, area, y = features(row)
    assert len(x) == len(ll) == len(y) and np.isfinite(x).all()
    assert (ll > 0).all() and (area > 0).all() and (y >= 0).all()
    report = dict(passed=True, reversed_edges=True, exact_exposure=True,
                  boundary_assignment=True, topology_prevents_branch_jump=True,
                  geometry_control_features=True, real_train_root=row['root_id'])
    save(ROOT / 'acceptance/geometry_contracts.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()

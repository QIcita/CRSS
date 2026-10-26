"""Prepare the original R38 computation views from one CPU fine-patch input.

This pure adapter does not read labels, files or experiment state. Call once
per cached sample, before the original tensor-to-device transfer. Existing
frozen R38 caches remain accepted without rebuilding their views.
"""
import numpy as np
import torch


def prepare_surface_inputs(geometry, patch_centers):
    """Retain every base field; reproduce C and dense-D inputs exactly.

    patch_centers must be the original cached patch_pos array, not grid-cell
    centres or recomputed high-precision source centres. geometry retains all
    point-level observations. No patch-summary inversion is attempted.
    """
    from tools.cpsf_raw_point_attention.data import grouping

    if {'label', 'target', 'split', 'observed', 'fine_target', 'coarse_target'}.intersection(geometry):
        raise ValueError('Pass geometry only; supervision does not belong in SurfaceSupport.')
    g = dict(geometry)
    # This adapter is the H01/FlyWire R38 path, not the TLP descriptor adapter.
    if 'raw_geometry_pf' in g:
        raise ValueError('Directional-feature inputs require their original task adapter.')
    center = np.asarray(patch_centers, dtype=np.float64)
    owner = g['point_owner'].numpy()
    if center.shape != (g['patches'], 3) or not len(owner):
        raise ValueError('Original nonempty point/patch geometry is required.')
    pos = center[owner] + g['pf'].numpy()[:, :3].astype(np.float64) * .5
    weight = g['point_weight'].numpy().astype(np.float64)

    # Hierarchy view: preserve the original origin, epsilon, sort and moments.
    origin = pos.min(0)
    grid = np.floor((pos - origin) / .5 + 1e-5).astype(np.int64)
    unique, inverse = np.unique(grid, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weight, minlength=len(unique))
    pf = g['pf'].numpy().astype(np.float64)
    mean = np.zeros((len(unique), 4)); second = mean.copy()
    coord = np.zeros((len(unique), 3))
    np.add.at(mean, inverse, pf * weight[:, None])
    np.add.at(second, inverse, pf ** 2 * weight[:, None])
    np.add.at(coord, inverse, (pos - origin) * weight[:, None])
    mean /= mass[:, None]; second /= mass[:, None]; coord /= mass[:, None]
    g.update(ucoord=torch.tensor(coord / .5, dtype=torch.float32),
             ugrid=torch.from_numpy(unique), uinverse=torch.from_numpy(inverse),
             ufeat=torch.tensor(np.column_stack((mean, (second - mean ** 2).clip(0), np.log1p(mass))), dtype=torch.float32))

    # Neighborhood view: original fine grouping, optionally dense computation.
    # Remove stale derived dense views if the caller reuses an input dictionary.
    for key in tuple(g):
        if key.startswith('dense_'):
            del g[key]
    g.update(grouping(owner, g['patches']))
    g['raw_patch_pos'] = torch.tensor(center, dtype=torch.float32) / .5
    g['patch_size'] = .5
    if len(owner) / g['patches'] < 10:
        for level in range(1, 25):
            size = .5 * 2 ** level
            _, dense_owner = np.unique(np.floor(pos / size).astype(np.int64), axis=0, return_inverse=True)
            if len(owner) / (dense_owner.max() + 1) >= 10:
                break
        else:
            raise RuntimeError('Finite scale search exhausted')
        dense_mass = np.bincount(dense_owner, weights=weight)
        centers = np.zeros((len(dense_mass), 3))
        np.add.at(centers, dense_owner, pos * weight[:, None]); centers /= dense_mass[:, None]
        relative = (pos - centers[dense_owner]) / size
        dense = dict(point_owner=torch.from_numpy(dense_owner),
                     patch_mass=torch.tensor(dense_mass, dtype=torch.float32),
                     patches=len(dense_mass), **grouping(dense_owner, len(dense_mass)))
        g.update({'dense_' + key: value for key, value in dense.items()},
                 dense_pf=torch.tensor(np.column_stack((relative, np.linalg.norm(relative, axis=1))), dtype=torch.float32) * (size / .5),
                 dense_raw_patch_pos=torch.tensor(centers / size, dtype=torch.float32) * (size / .5),
                 dense_size=size)
    return g

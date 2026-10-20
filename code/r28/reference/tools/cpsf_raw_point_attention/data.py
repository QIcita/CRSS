"""Unchanged R9 observations, with lossless ragged within-patch grouping."""
import numpy as np
import torch
from tools.cpsf_utonia_context.data import load_item as original, move


def grouping(owner, patches, slot_budget=16384):
    owner = np.asarray(owner, dtype=np.int64)
    counts = np.bincount(owner, minlength=patches)
    order = np.argsort(owner, kind='stable')
    ptr = np.r_[0, np.cumsum(counts)]
    layouts, patches_out, indices, valid_out = [], [], [], []
    npatch = nslot = 0
    # Singleton patches retain exactly the original R9 path.
    widths = np.where(counts > 1, 2 ** np.ceil(np.log2(np.maximum(counts, 1))).astype(np.int64), 0)
    for width in np.unique(widths[widths > 0]):
        ids = np.flatnonzero(widths == width)
        size = max(1, slot_budget // int(width))
        for lo in range(0, len(ids), size):
            rows = ids[lo:lo + size]
            ix = np.zeros((len(rows), width), dtype=np.int64)
            valid = np.zeros_like(ix, dtype=bool)
            for i, patch in enumerate(rows):
                values = order[ptr[patch]:ptr[patch + 1]]
                ix[i, :len(values)] = values
                valid[i, :len(values)] = True
            layouts.append((npatch, npatch + len(rows), nslot, nslot + ix.size, int(width)))
            npatch += len(rows); nslot += ix.size
            patches_out.append(rows); indices.append(ix.ravel()); valid_out.append(valid.ravel())
    cat = lambda xs, dtype: np.concatenate(xs).astype(dtype) if xs else np.empty(0, dtype=dtype)
    return dict(raw_patch_ids=torch.from_numpy(cat(patches_out, np.int64)),
                raw_indices=torch.from_numpy(cat(indices, np.int64)),
                raw_valid=torch.from_numpy(cat(valid_out, bool)),
                raw_counts=torch.from_numpy(counts), raw_layout=layouts)


def load_item(row):
    g = original(row)
    g.update(grouping(g['point_owner'].numpy(), g['patches']))
    with np.load(row['feature_path']) as a:
        g['raw_patch_pos'] = torch.tensor(a['patch_pos'], dtype=torch.float32) / .5
    return g

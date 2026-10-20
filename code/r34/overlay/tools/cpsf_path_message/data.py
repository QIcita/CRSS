"""Original R28 tensors, plus complete incidence batching for attention only."""
import numpy as np
import torch
from tools.cpsf_raw_point_attention.data import load_item as parent, move


def load_item(row):
    g = parent(row)
    pr = g['pr'].numpy()
    counts = np.bincount(pr, minlength=g['nregions'])
    order = np.argsort(pr, kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    matched = (g['has_g'] & g['has_p']).numpy()
    width = np.where(counts > 0, 2 ** np.ceil(np.log2(np.maximum(counts, 1))).astype(np.int64), 0)
    layouts, regions, indices, valid_all = [], [], [], []
    nregion = nslot = 0
    for size in np.unique(width[matched & (counts > 0)]):
        ids = np.flatnonzero(matched & (width == size))
        group = max(1, 16384 // int(size))
        for left in range(0, len(ids), group):
            rr = ids[left:left+group]
            ix = np.zeros((len(rr), size), dtype=np.int64)
            valid = np.zeros_like(ix, dtype=bool)
            for i, r in enumerate(rr):
                values = order[starts[r]:starts[r+1]]
                ix[i, :len(values)] = values
                valid[i, :len(values)] = True
            layouts.append((nregion, nregion+len(rr), nslot, nslot+ix.size, int(size)))
            regions.append(rr); indices.append(ix.ravel()); valid_all.append(valid.ravel())
            nregion += len(rr); nslot += ix.size
    cat = lambda xs, dtype: np.concatenate(xs).astype(dtype) if xs else np.empty(0, dtype=dtype)
    g.update(att_layout=layouts, att_regions=torch.from_numpy(cat(regions, np.int64)),
             att_indices=torch.from_numpy(cat(indices, np.int64)),
             att_valid=torch.from_numpy(cat(valid_all, bool)))
    return g

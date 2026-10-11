"""Class-balanced sample objective, compatible with per-cell accumulation."""
import torch
from torch.nn import functional as F


def weights_from_counts(counts):
    n = torch.as_tensor(counts, dtype=torch.float64)
    assert n.ndim == 1 and (n > 0).all()
    return (n.sum() / (len(n) * n)).float()


def balanced_loss(logits, target, weights, smoothing=.05):
    # Do NOT use CE(weight=w, reduction='mean') on one cell: its denominator
    # cancels the target weight. Normalize accumulation by cell count instead.
    per_cell = F.cross_entropy(logits, target, label_smoothing=smoothing, reduction='none')
    return (per_cell * weights[target]).mean()

import torch
from torch.nn import functional as F


def missing_edge_loss(logits, target, observed):
    """All unobserved unordered pairs; no candidate/GT-derived input support."""
    assert logits.shape == target.shape == observed.shape
    assert logits.ndim == 2 and logits.shape[0] == logits.shape[1]
    mask = torch.triu(~observed.bool(), diagonal=1)
    scores, labels = logits[mask], target[mask]
    assert scores.numel() > 0, 'No unobserved pair exists'
    positive = labels.sum()
    negative = labels.numel() - positive
    # Degenerate all-negative/all-positive graphs retain ordinary BCE.
    positive_weight = torch.where((positive > 0) & (negative > 0),
                                  negative / positive.clamp_min(1),
                                  torch.ones_like(positive))
    weight = torch.where(labels > 0, positive_weight, torch.ones_like(labels))
    return F.binary_cross_entropy_with_logits(scores, labels, weight=weight)

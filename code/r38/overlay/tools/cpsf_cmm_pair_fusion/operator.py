"""Bounded pair interactions at the original 96D CMM combination interface."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PAIRS = ((0, 1), (0, 2), (1, 2))
WEIGHTS = (1., .1, .1)


class PairFusion(nn.Module):
    rank = 16
    alpha = .1
    epsilon = 1e-6
    chunk = 4096
    checkpointed = True
    enabled = True

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(96, self.rank, bias=False) for _ in range(3)])
        self.outputs = nn.ParameterList([nn.Parameter(torch.zeros(96, self.rank)) for _ in range(3)])

    def terms(self, h, d, c):
        xs = (h, d, c)
        z = [torch.tanh(a(x)) for a, x in zip(self.projections, xs)]
        ns = [((w*x).square().sum(-1, keepdim=True) + self.epsilon**2).sqrt() - self.epsilon
              for x, w in zip(xs, WEIGHTS)]
        result = []
        for (i, j), b in zip(PAIRS, self.outputs):
            weight = b / (1 + b.square().sum()).sqrt()
            value = z[i] * z[j] / math.sqrt(self.rank)
            result.append(self.alpha / 2 * (ns[i] + ns[j]) * F.linear(value, weight))
        return tuple(result)

    def block(self, h, d, c):
        a, b, e = self.terms(h, d, c)
        return a + b + e

    def forward(self, h, d, c):
        assert h.shape == d.shape == c.shape and h.ndim == 2 and h.shape[-1] == 96
        # Parent worker uses FP32. Do not silently run the bound in half precision.
        assert h.dtype in (torch.float32, torch.float64)
        if not self.enabled:
            return torch.zeros_like(h)
        pieces = []
        for left in range(0, len(h), self.chunk):
            args = (h[left:left+self.chunk], d[left:left+self.chunk], c[left:left+self.chunk])
            pieces.append(checkpoint(self.block, *args, use_reentrant=False, preserve_rng_state=True)
                          if self.checkpointed and self.training and torch.is_grad_enabled()
                          else self.block(*args))
        return torch.cat(pieces) if pieces else torch.zeros_like(h)


def gates(model):
    return [model.cmm_pair_fusion]


def set_enabled(model, enabled):
    model.cmm_pair_fusion.enabled = enabled

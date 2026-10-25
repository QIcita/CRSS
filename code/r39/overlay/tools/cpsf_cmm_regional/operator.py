"""A single sparse integration operator; normalization affects routing, not values."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def add(x, index, size):
    return x.new_zeros((size,) + x.shape[1:]).index_add(0, index, x)


class RegionalIntegration(nn.Module):
    width = 96
    heads = 4
    key_width = 8
    chunk = 8192
    checkpointed = True

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(96, 32, bias=False)
        self.keys = nn.ModuleList([nn.Linear(96, 32, bias=False) for _ in range(3)])
        self.geometry = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32))
        self.fuse = nn.Linear(4 * 96, 96, bias=False)
        # A trainable full projection, initialized to a conservative mixture.
        # These coefficients are initialization, not a fixed fusion rule.
        with torch.no_grad():
            self.fuse.weight.zero_()
            for i, scale in enumerate((.5, .5, .05, .05)):
                self.fuse.weight[:, i*96:(i+1)*96].copy_(torch.eye(96) * scale)

    def edge_weights(self, query, key, phi, mass, region, patch):
        geom = self.geometry(phi).reshape(-1, 4, 8)
        score = (query[region] * (key[patch] + geom)).sum(-1) / math.sqrt(8)
        # Bounded scores allow a stable streaming reduction without a dense mask.
        return mass[:, None] * (2 * torch.tanh(score / 2)).exp()

    def integrate(self, graph, features, key_module, g):
        regions = g['nregions']
        query = self.query(F.layer_norm(graph, (96,))).reshape(-1, 4, 8)
        key = key_module(F.layer_norm(features, (96,))).reshape(-1, 4, 8)
        value = features.reshape(-1, 4, 24)
        total = features.new_zeros((regions, 4, 24))
        attended = torch.zeros_like(total)
        normalizer = features.new_zeros((regions, 4))
        measure = features.new_zeros((regions,))

        def block(q, k, v, phi, mass, r, j):
            a = self.edge_weights(q, k, phi, mass, r, j)
            return (add(v[j] * mass[:, None, None], r, regions),
                    add(v[j] * a[:, :, None], r, regions),
                    add(a, r, regions), add(mass, r, regions))

        for left in range(0, len(g['pr']), self.chunk):
            sl = slice(left, left + self.chunk)
            r, j = g['pr'][sl], g['pj'][sl]
            mass = g['pw'][sl] * g['patch_mass'][j]
            args = (query, key, value, g['phi'][sl], mass, r, j)
            a, b, c, d = (checkpoint(block, *args, use_reentrant=False)
                           if self.checkpointed and self.training and torch.is_grad_enabled()
                           else block(*args))
            total, attended = total + a, attended + b
            normalizer, measure = normalizer + c, measure + d
        mean = total / measure.clamp_min(1e-12)[:, None, None]
        adaptive = attended / normalizer.clamp_min(1e-12)[:, :, None]
        # Each observed incidence retains at least half its normalized mass.
        return (.5 * (mean + adaptive)).reshape(regions, 96)

    def forward(self, graph, h, d, c, g):
        parts = [self.integrate(graph, x, key, g) for x, key in zip((h, d, c), self.keys)]
        graph = graph * g['has_g'][:, None]
        parts = [x * g['has_p'][:, None] for x in parts]
        present = (g['has_g'].to(graph.dtype) + g['has_p'].to(graph.dtype)).clamp_min(1)
        return self.fuse(torch.cat([graph] + parts, -1) * (2 / present[:, None]))

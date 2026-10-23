"""R34: change only the observed patch-to-region message in frozen R28.

Project gated messages are not an official model reproduction. The attention
control retains the R17 official-GeoTransformer-derived math and mass adapter.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tools.cpsf_raw_point_attention.model import Model as Parent
from tools.cpsf_v21_segment_units.model import add

VARIANTS = ('path_message', 'constant_message', 'ordinary_attention')


def surface_mean(patch, g, chunk=16384):
    def integrate(features):
        result = features.new_zeros((g['nregions'], 96))
        for left in range(0, len(g['pr']), chunk):
            sl = slice(left, left + chunk)
            r, j = g['pr'][sl], g['pj'][sl]
            w = g['pw'][sl] * g['patch_mass'][j]
            result = result + add(features[j] * w[:, None], r, g['nregions'])
        return result / g['surface_mass'].clamp_min(1e-12)[:, None]
    return checkpoint(integrate, patch, use_reentrant=False) if torch.is_grad_enabled() else integrate(patch)


class PathMessage(nn.Module):
    chunk = 16384
    checkpointed = True

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(96, 16, bias=False)
        self.key = nn.Linear(96, 16, bias=False)
        self.value = nn.Linear(96, 16, bias=False)
        self.geometry = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 16))
        # Both arms have the same effective constant and projections. Removing
        # observed graph content does not replace a matrix with a smaller model.
        self.constant = nn.Parameter(torch.zeros(96))
        self.output = nn.Linear(16, 96, bias=False)
        nn.init.zeros_(self.output.weight)

    def features(self, graph, patch, g, kind):
        mu = surface_mean(patch, g, self.chunk)
        content = F.layer_norm(graph, (96,)) if kind == 'path_message' else torch.zeros_like(graph)
        query = self.query(content + self.constant)
        key = self.key(F.layer_norm(patch, (96,)))
        result = patch.new_zeros((g['nregions'], 16))
        for left in range(0, len(g['pr']), self.chunk):
            right = min(left + self.chunk, len(g['pr']))
            def block(query, key, patch, mu, left=left, right=right):
                r, j = g['pr'][left:right], g['pj'][left:right]
                w = g['pw'][left:right] * g['patch_mass'][j]
                w = w / g['surface_mass'][r].clamp_min(1e-12)
                gate = torch.tanh(query[r] + key[j] + self.geometry(g['phi'][left:right]))
                message = gate * self.value(patch[j] - mu[r])
                return add(message * w[:, None], r, g['nregions'])
            args = (query, key, patch, mu)
            result = result + (checkpoint(block, *args, use_reentrant=False)
                               if self.checkpointed and torch.is_grad_enabled() else block(*args))
        return result

    def forward(self, graph, patch, g, kind):
        return self.output(self.features(graph, patch, g, kind))


class OrdinaryAttention(nn.Module):
    checkpointed = True

    def __init__(self):
        super().__init__()
        from .attention import MassAttention
        self.attention = MassAttention(96, 4, dropout=None)
        self.attention.proj_v.register_parameter('bias', None)
        self.geometry = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        self.output = nn.Linear(96, 96, bias=False)
        nn.init.zeros_(self.output.weight)

    def features(self, graph, patch, g, kind):
        mu = surface_mean(patch, g)
        result = patch.new_zeros((g['nregions'], 96))
        for lo, hi, left, right, width in g['att_layout']:
            def block(graph, patch, mu, lo=lo, hi=hi, left=left, right=right, width=width):
                ids = g['att_regions'][lo:hi]
                ix = g['att_indices'][left:right].reshape(hi-lo, width)
                valid = g['att_valid'][left:right].reshape(hi-lo, width)
                j = g['pj'][ix]
                p = patch[j]
                q = F.layer_norm(graph[ids], (96,))[:, None]
                k = F.layer_norm(p, (96,))
                value = p - mu[ids, None]
                w = g['pw'][ix] * g['patch_mass'][j]
                bias = self.geometry(g['phi'][ix]).squeeze(-1)[:, None]
                z, _ = self.attention(q, k, value, key_weights=w,
                                      key_masks=~valid, attention_factors=bias)
                return z[:, 0]
            args = (graph, patch, mu)
            z = checkpoint(block, *args, use_reentrant=False) if self.checkpointed and torch.is_grad_enabled() else block(*args)
            result = result.index_add(0, g['att_regions'][lo:hi], z)
        return result

    def forward(self, graph, patch, g, kind):
        return self.output(self.features(graph, patch, g, kind))


class Model(Parent):
    def __init__(self, task, kind):
        assert kind in VARIANTS + ('r28_control',)
        super().__init__(task, 'r9_lpa_attention')
        self.strategy = kind
        self.message_enabled = kind != 'r28_control'
        state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                np.random.seed(42)
                self.region_message = OrdinaryAttention() if kind == 'ordinary_attention' else PathMessage()
        finally:
            np.random.set_state(state)

    def patch_context_delta(self, hp, pooled):
        original = super().patch_context_delta(hp, pooled)
        self._message_patch = hp + original
        return original

    def stems(self, g):
        def capture(module, args, value):
            self._message_graph = value
        handle = self.graph.register_forward_hook(capture)
        try:
            graph, point = super().stems(g)
            if not self.message_enabled:
                return graph, point
            sr, sn = g['sr'], g['sn']
            path = (g['s'][sn] - g['pos'][sr]) / g['radius']
            raw_graph = add((self._message_graph[sn] + self.path_position(path)) * g['sw'][:, None],
                            sr, g['nregions']) / g['graph_mass'].clamp_min(1e-12)[:, None]
            delta = self.region_message(raw_graph, self._message_patch, g, self.strategy)
            delta = delta * (g['has_g'] & g['has_p'])[:, None]
            # Addition after the original shared code is explicit; the original
            # R28 stem arithmetic is otherwise byte-for-byte inherited.
            return graph, point + .1 * delta
        finally:
            handle.remove()
            for name in ('_message_graph', '_message_patch'):
                if hasattr(self, name):
                    delattr(self, name)

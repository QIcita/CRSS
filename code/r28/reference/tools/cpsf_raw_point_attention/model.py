"""Refine raw within-patch evidence; protected R9 statistics/context stay intact."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tools.cpsf_utonia_context.model import H01 as R9
from tools.cpsf_v21_segment_units.model import add
from .official import build

VARIANTS = ('r9_lpa_attention', 'r9_lpa_uniform')


class RawPointRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = {}
        self.local = build(self.context)
        self.output = nn.Linear(96, 96, bias=False)
        nn.init.zeros_(self.output.weight)

    def descriptors(self, px, g, kind, checkpointed=True):
        mu = add(px * g['point_weight'][:, None], g['point_owner'], g['patches']) / g['patch_mass'].clamp_min(1e-12)[:, None]
        output = []
        for lo, hi, left, right, width in g['raw_layout']:
            # Bind loop-local values for backward checkpoint recomputation.
            def block(features, mean, lo=lo, hi=hi, left=left, right=right, width=width):
                ids = g['raw_patch_ids'][lo:hi]
                ix = g['raw_indices'][left:right].reshape(hi-lo, width)
                valid = g['raw_valid'][left:right].reshape(hi-lo, width)
                logmass = g['point_weight'][ix].clamp_min(1e-12).log().masked_fill(~valid, -torch.inf)
                self.context.update(kind=kind, log_mass=logmass[None, None])
                self.local.grouper.relative_xyz = g['pf'][ix, :3].permute(2, 0, 1)[None]
                self.local.grouper.raw_features = features[ix].permute(2, 0, 1)[None]
                try:
                    _, refined = self.local((g['raw_patch_pos'][ids][None], mean[ids].T[None]))
                    return refined[0].T - mean[ids]
                finally:
                    self.context.clear()
                    del self.local.grouper.relative_xyz, self.local.grouper.raw_features
            output.append(checkpoint(block, px, mu, use_reentrant=False, preserve_rng_state=True)
                          if checkpointed and torch.is_grad_enabled() else block(px, mu))
        result = px.new_zeros((g['patches'], 96))
        if output:
            result = result.index_add(0, g['raw_patch_ids'], torch.cat(output))
        return result

    def forward(self, px, g, kind):
        return self.output(self.descriptors(px, g, kind))


class Model(R9):
    def __init__(self, task, kind):
        assert task in ('h01_local', 'h01_global') and kind in VARIANTS + ('r9_control',)
        super().__init__(task, 'utonia')
        self.adapter_variant = 'patch_rms_010'; self.global_enabled = False; self.kind = kind
        import numpy as np
        state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                np.random.seed(42)
                self.raw_refine = RawPointRefinement()
        finally:
            np.random.set_state(state)

    def patch_context_delta(self, hp, pooled):
        original = super().patch_context_delta(hp, pooled)
        if self.kind == 'r9_control':
            return original
        return original + .1 * self.raw_refine(self._raw_px, self._raw_g, self.kind)

    def stems(self, g):
        self._raw_g = g
        def capture(module, args, value):
            self._raw_px = value
        handle = self.point.register_forward_hook(capture)
        try:
            return super().stems(g)
        finally:
            handle.remove()
            del self._raw_g, self._raw_px

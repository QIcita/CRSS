"""Attach fusion after the exact inherited CMM paths, before both consumers."""
import torch
from .common import add_backend
from .operator import PairFusion
add_backend()
from tools.cpsf_r34_fps_rebench.model import H01Model as ParentH01


class H01Fusion(ParentH01):
    def patch_context_delta(self, hp, pooled):
        if not self.cmm_pair_fusion.enabled:
            return super().patch_context_delta(hp, pooled)
        local = []
        handle = self.raw_refine.register_forward_hook(lambda module, args, value: local.append(value))
        try:
            original = super().patch_context_delta(hp, pooled)
        finally:
            handle.remove()
        assert len(local) == 1
        # Preserve original RMS0.1 arithmetic above. RMS1.0 is used only by
        # the added relation branch, so the original pathway stays exact.
        calibrated = self.rms_delta(hp, pooled, 1.)
        delta = self.cmm_pair_fusion(hp, local[0], calibrated)
        combined = original + delta
        # Both the original surface integration and R34 PathMessage must
        # consume the same patch representation.
        self._message_patch = hp + combined
        return combined


def install(model):
    assert type(model) is ParentH01
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        fusion = PairFusion()
    fusion = fusion.to(next(model.parameters()))
    model.__class__ = H01Fusion
    model.add_module('cmm_pair_fusion', fusion)
    assert sum(p.numel() for p in fusion.parameters()) == 9216
    return model

import torch
from .data import DENSE_FIELDS
from tools.cpsf_cmm_pair_fusion.operator import PairFusion
from tools.cpsf_r34_fps_rebench.model import H01Model,FlyModel
from tools.cpsf_raw_point_attention.model import RawPointRefinement
from tools.cpsf_v21_segment_units.model import add

class DenseRefinement(RawPointRefinement):
    def forward(self,px,g,kind):
        if 'dense_size' not in g:return super().forward(px,g,kind)
        dense={k:g['dense_'+k] for k in DENSE_FIELDS}
        dense.update(pf=g['dense_pf'],raw_patch_pos=g['dense_raw_patch_pos'],point_weight=g['point_weight'])
        delta=super().forward(px,dense,kind)
        return add(delta[dense['point_owner']]*g['point_weight'][:,None],g['point_owner'],g['patches'])/g['patch_mass'].clamp_min(1e-12)[:,None]

class PairMixin:
    # Identical R38 fusion arithmetic, shared by the original H01/FlyWire heads.
    def patch_context_delta(self,hp,pooled):
        if not self.cmm_pair_fusion.enabled:return super().patch_context_delta(hp,pooled)
        local=[]
        handle=self.raw_refine.register_forward_hook(lambda module,args,value:local.append(value))
        try:original=super().patch_context_delta(hp,pooled)
        finally:handle.remove()
        assert len(local)==1
        calibrated=self.rms_delta(hp,pooled,1.)
        delta=self.cmm_pair_fusion(hp,local[0],calibrated)
        combined=original+delta
        self._message_patch=hp+combined
        return combined

class H01Dense(PairMixin,H01Model):pass
class FlyDense(PairMixin,FlyModel):pass

def install(model):
    assert type(model) in (H01Model,FlyModel)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42);fusion=PairFusion()
    fusion=fusion.to(next(model.parameters()))
    model.__class__=H01Dense if type(model) is H01Model else FlyDense
    model.raw_refine.__class__=DenseRefinement
    model.add_module('cmm_pair_fusion',fusion)
    assert sum(p.numel() for p in fusion.parameters())==9216
    return model

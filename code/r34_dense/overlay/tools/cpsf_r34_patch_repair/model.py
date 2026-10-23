import torch
from .common import variant
from .data import DENSE_FIELDS
from tools.cpsf_r34_fps_rebench.model import H01Model
from tools.cpsf_r34_adaptive_patch.model import PhysicalPatchScale
from tools.cpsf_utonia_context.model import Integration
from tools.cpsf_v21_segment_units.model import add

class ContextControl(PhysicalPatchScale,H01Model):pass

class SupportPreserved(H01Model):
    def patch_context_delta(self,hp,pooled):
        g=self._raw_g
        if 'dense_size' not in g:return super().patch_context_delta(hp,pooled)
        # Existing RMS0.1 Utonia integration, original fine local statistics,
        # original PathMessage and original model parameter/initialization stream.
        context=Integration.patch_context_delta(self,hp,pooled)
        dense={k:g['dense_'+k] for k in DENSE_FIELDS}
        dense.update(pf=g['dense_pf'],raw_patch_pos=g['dense_raw_patch_pos'],point_weight=g['point_weight'])
        delta=self.raw_refine(self._raw_px,dense,'r9_lpa_attention')
        # Scatter through actual point membership, not a patch-centroid query.
        # Every original point contributes its original mass exactly once.
        lifted=add(delta[dense['point_owner']]*g['point_weight'][:,None],
                   g['point_owner'],g['patches'])/g['patch_mass'].clamp_min(1e-12)[:,None]
        result=context+.1*lifted
        self._message_patch=hp+result
        return result

def Model(task):
    assert task=='h01_global'
    return (ContextControl if variant()=='context_control' else SupportPreserved)(task)

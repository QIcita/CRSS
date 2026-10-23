"""The identical R28 raw-point module on inherited task heads; scratch per task."""
import numpy as np
import torch
from tools.cpsf_utonia_context.model import Integration, H01, TED
from tools.cpsf_fixed_core.model import FlyModel
from tools.cpsf_r9_expansion.model import TLP as R9TLP
from tools.cpsf_raw_point_attention.model import RawPointRefinement

class RawMixin:
    def configure_raw(self):
        self.adapter_variant='patch_rms_010';self.global_enabled=False
        state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42);np.random.seed(42)
                self.raw_refine=RawPointRefinement()
        finally:np.random.set_state(state)
    def patch_context_delta(self,hp,pooled):
        original=super().patch_context_delta(hp,pooled)
        # TLP content uses directional XYZ; LPAMLP's geometry retains observed
        # within-patch XYZ, just as all other tasks. No new points or sampling.
        geom=dict(self._raw_g,pf=self._raw_g['raw_geometry_pf']) if 'raw_geometry_pf' in self._raw_g else self._raw_g
        return original+.1*self.raw_refine(self._raw_px,geom,'r9_lpa_attention')
    def stems(self,g):
        self._raw_g=g
        def capture(module,args,value):self._raw_px=value
        handle=self.point.register_forward_hook(capture)
        try:return super().stems(g)
        finally:
            handle.remove();del self._raw_g,self._raw_px

class Fly(RawMixin,Integration,FlyModel):
    def __init__(self):
        super().__init__('flywire_coarse','full');self.configure('utonia');self.configure_raw()
class TLP(RawMixin,R9TLP):
    def __init__(self):super().__init__();self.configure_raw()
class Ted(RawMixin,TED):
    def __init__(self):super().__init__('ted','utonia');self.configure_raw()
class H01Check(RawMixin,H01):
    def __init__(self,task):super().__init__(task,'utonia');self.configure_raw()
def Model(task):
    return H01Check(task) if task.startswith('h01_') else Fly() if task=='flywire_coarse' else Ted() if task=='ted' else TLP()

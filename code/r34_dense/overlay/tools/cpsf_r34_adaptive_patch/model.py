import math
from .common import REFERENCE
from tools.cpsf_r34_fps_rebench.model import H01Model as ParentH01,FlyModel as ParentFly

class PhysicalPatchScale:
    def stems(self,g):
        h=float(g['patch_size'])
        if h==.5:return super().stems(g)
        def actual_scale(module,args):
            x=args[0].clone();x[:,2]=math.log1p(h)
            return (x,)+args[1:]
        handle=self.scale.register_forward_pre_hook(actual_scale)
        try:return super().stems(g)
        finally:handle.remove()

class H01(PhysicalPatchScale,ParentH01):pass
class Fly(PhysicalPatchScale,ParentFly):pass

def Model(task):return H01(task) if task=='h01_global' else Fly()

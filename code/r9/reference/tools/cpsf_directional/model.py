import torch
from tools.cpsf_deterministic.model import Model as Base

class Model(Base):
    def __init__(self,task,arm,descriptor='voxel'):
        assert descriptor in ('voxel','directional_xyz','directional')
        super().__init__(task,arm,'voxel');self.directional_descriptor=descriptor
    def stems(self,g):
        if self.arm=='skeleton' or self.directional_descriptor=='voxel':return super().stems(g)
        pf=g['pf_directional']
        if self.directional_descriptor=='directional_xyz':pf=torch.cat((pf[:,:3],g['pf'][:,3:]),1)
        return super().stems(dict(g,pf=pf))

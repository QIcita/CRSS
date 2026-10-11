import torch
from tools.cpsf_deterministic.model import Model as Base

DESCRIPTORS=('voxel','shape_only','extent_only','neighborhood')

def descriptor(pf,neighborhood,kind):
    assert kind in DESCRIPTORS and pf.shape==neighborhood.shape
    if kind=='voxel':return pf
    if kind=='neighborhood':return neighborhood
    if kind=='shape_only':return torch.cat((neighborhood[:,:3],pf[:,3:4]),1)
    return torch.cat((pf[:,:3],neighborhood[:,3:4]),1)

class Model(Base):
    def __init__(self,task,arm,descriptor='voxel'):
        assert descriptor in DESCRIPTORS
        super().__init__(task,arm,'voxel')
        self.factor_descriptor=descriptor

    def stems(self,g):
        if self.arm=='skeleton' or self.factor_descriptor=='voxel':return super().stems(g)
        pf=descriptor(g['pf'],g['pf_neighborhood'],self.factor_descriptor)
        return super().stems(dict(g,pf=pf))

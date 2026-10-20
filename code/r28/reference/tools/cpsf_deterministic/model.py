import torch
from tools.cpsf_point_geometry.model import Model as Base

def weighted_region_sum(x,row,col,weight,count):
    """Same fixed incidence product; deterministic gather and index reduction."""
    assert torch.are_deterministic_algorithms_enabled()
    assert not weight.requires_grad
    return x.new_zeros((count,x.shape[1])).index_add(0,row,x[col]*weight[:,None])

class Model(Base):
    def __init__(self,task,arm,descriptor='neighborhood'):
        assert descriptor in ('voxel','neighborhood')
        super().__init__(task,arm)
        self.use_descriptor=descriptor=='neighborhood'

    def corridor_pool(self,x,g,mass,active):
        row,col=g['bridge_row'],g['bridge_col'];count=g['pair_count']
        weight=g['bridge_weight']*mass[col]*active[col]
        den=x.new_zeros(count).index_add(0,row,weight)
        weight=weight/den[row].clamp_min(1e-12)
        return weighted_region_sum(x,row,col,weight,count),den>0

"""Unchanged V23 encoder with a classification head and 128D retrieval feature."""
import torch
from torch import nn
from tools.cpsf_v21_segment_units.model import add
from torch.utils.checkpoint import checkpoint
from tools.cpsf_v23_scale_ablation.model import Model as Encoder

ARMS=('full','mean_only','no_fusion','point','skeleton','no_position','no_scale',
      'no_position_scale','thumbnail','uniform_budget')
VARIANTS=('joint','point','skeleton','thumbnail','uniform_budget')

def variant(arm):return arm if arm in VARIANTS else 'joint'

class Model(Encoder):
    def __init__(self,classes,arm):
        assert arm in ARMS
        mode=arm if arm in ('no_position','no_scale','no_position_scale','thumbnail','uniform_budget') else 'full'
        super().__init__('global',mode)
        self.arm=arm if arm in ('full','mean_only','no_fusion','point','skeleton') else 'full'
        self.experiment_arm=arm
        self.head=nn.Linear(128,classes)

    chunk_size=32768

    def reduce_surface(self,hp,phi,pr,pj,wp,u):
        return add((hp[pj]+self.relation(phi))*wp[:,None],pr,u)

    def stems(self,g):
        if len(g['pr'])>self.chunk_size:
            return self.chunked_stems(g)
        if self.training and torch.is_grad_enabled():
            return checkpoint(super().stems,g,use_reentrant=False)
        return super().stems(g)

    def chunked_stems(self,g):
        u=g['nregions'];hg=self.graph(g['sf']);px=self.point(g['pf'])
        den=g['patch_mass'].clamp_min(1e-12)[:,None]
        mu=add(px*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        second=add(px.square()*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        hp=self.patch(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))
        sr,sn=g['sr'],g['sn'];pr,pj=g['pr'],g['pj']
        path=(g['s'][sn]-g['pos'][sr])/g['radius']
        path_code=self.path_position(path) if self.use_position else torch.zeros_like(hg[sn])
        graph=add((hg[sn]+path_code)*g['sw'][:,None],sr,u)/g['graph_mass'].clamp_min(1e-12)[:,None]
        wp=g['pw']*g['patch_mass'][pj]
        surface=hp.new_zeros((u,hp.shape[1]))
        for lo in range(0,len(pr),self.chunk_size):
            hi=lo+self.chunk_size
            args=(hp,g['phi'][lo:hi],pr[lo:hi],pj[lo:hi],wp[lo:hi])
            if self.training and torch.is_grad_enabled():
                value=checkpoint(self.reduce_surface,*args,u,use_reentrant=False)
            else:value=self.reduce_surface(*args,u)
            surface=surface+value
        surface=surface/g['surface_mass'].clamp_min(1e-12)[:,None]
        # Bounded local position; true physical length and resolution stay explicit.
        xyz=g['pos'];xyz=xyz-xyz.mean(0)
        position=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius'])) if self.use_position else torch.zeros_like(graph)
        scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
        code=position+(self.scale(scale) if self.use_scale else torch.zeros_like(graph))
        return (graph+code)*g['has_g'][:,None],(surface+code)*g['has_p'][:,None]


    def embedding(self,g):
        graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass']
        wg=gm/gm.sum().clamp_min(1e-12);wp=pm/pm.sum().clamp_min(1e-12)
        if self.arm in ('full','mean_only'):
            active=g['has_g']|g['has_p'];mass=wg+wp
            x=self.combine(graph,point,g['has_g'],g['has_p'])
            z=self.pool(self.processor(x,g,mass,active),g,mass,active)
        else:
            streams=[]
            for x,mass,active in ((graph,wg,g['has_g']),(point,wp,g['has_p'])):
                if active.any():streams.append(self.pool(self.processor(x,g,mass,active),g,mass,active))
            assert streams;z=torch.stack(streams).mean(0)
        return self.readout(z)

    def forward(self,g):return self.head(self.embedding(g))

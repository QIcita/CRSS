import torch
from torch import nn
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.model import add
from tools.cpsf_v23_scale_ablation.model import Model as H01Base
from tools.cpsf_flywire_v23.model import Model as FlyBase

class GateMixin:
    def init_gate(self,arm):
        self.variant=arm
        self.use_position=arm not in ('no_position','no_position_scale')
        self.use_scale=arm not in ('no_scale','no_position_scale')
        self.processor.use_position=self.use_position
        with torch.random.fork_rng(devices=[]):
            self.region_gate=nn.Linear(98,1)
        nn.init.zeros_(self.region_gate.weight);nn.init.zeros_(self.region_gate.bias)
        self.gate_alpha=nn.Parameter(torch.zeros(()))

    def quality(self,g):
        # Region-mean normalized distance and patch association concentration.
        u=g['nregions'];w=g['pw']*g['patch_mass'][g['pj']]
        den=g['surface_mass'].clamp_min(1e-12)
        dist=add(w*g['phi'][:,0],g['pr'],u)/den
        concentration=add(w*g['pw'],g['pr'],u)/den
        return torch.stack((dist,concentration),-1)

    def gate_values(self,graph,point,g):
        if self.variant=='gate_global':regional=graph.new_full((len(graph),),.5)
        else:
            # Difference cancels the shared additive physical code at matched regions.
            diff=F.layer_norm(graph-point,(graph.shape[-1],))
            regional=self.region_gate(torch.cat((diff,self.quality(g)),-1)).squeeze(-1).sigmoid()
        return .1*self.gate_alpha.tanh()*regional

    def embedding(self,g):
        graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass']
        wg=gm/gm.sum().clamp_min(1e-12);wp=pm/pm.sum().clamp_min(1e-12)
        arm=self.variant
        if arm in ('no_fusion','point','skeleton'):
            streams=[]
            for x,mass,active in ((graph,wg,g['has_g']),(point,wp,g['has_p'])):
                if active.any():streams.append(self.pool(self.processor(x,g,mass,active),g,mass,active))
            z=torch.stack(streams).mean(0)
        else:
            active=g['has_g']|g['has_p'];mass=wg+wp
            count=(g['has_g'].to(graph.dtype)+g['has_p'].to(graph.dtype)).clamp_min(1)
            x=(graph+point)/count[:,None]
            matched=g['has_g']&g['has_p']
            if arm!='mean_only' and matched.any():
                delta=torch.zeros_like(x);delta[matched]=self.joint(torch.cat((graph[matched],point[matched]),-1))
                gain=.1 if arm=='fixed_residual' else self.gate_values(graph,point,g)[:,None]
                x=x+gain*delta
            z=self.pool(self.processor(x,g,mass,active),g,mass,active)
        return self.readout(z)

    def forward(self,g):
        y=self.head(self.embedding(g))
        return y.sigmoid().squeeze(-1) if self.dataset_task=='h01_global' else y

class H01Model(GateMixin,H01Base):
    def __init__(self,task,arm):
        super().__init__('local' if task=='h01_local' else 'global','full')
        self.dataset_task=task;self.init_gate(arm)

class FlyModel(GateMixin,FlyBase):
    def __init__(self,task,arm):
        super().__init__(6,'full');self.dataset_task=task;self.init_gate(arm)

def Model(task,arm):return FlyModel(task,arm) if task=='flywire_coarse' else H01Model(task,arm)

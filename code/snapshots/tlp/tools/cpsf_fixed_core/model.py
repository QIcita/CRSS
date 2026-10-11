import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.model import add
from tools.cpsf_v23_scale_ablation.model import Model as H01Base,Processor as BaseProcessor
from tools.cpsf_flywire_v23.model import Model as FlyBase

class Processor(BaseProcessor):
    use_context=True
    def forward(self,x,g,mass,active):
        if self.use_context:return super().forward(x,g,mass,active)
        # Exact same local propagation and FFN; skip only the coarse attention residual.
        x=x*active[:,None];e=g['adjacency'];e=e[active[e[:,0]]&active[e[:,1]]]
        a,b=e.unbind(1);r=torch.cat((a,b));j=torch.cat((b,a))
        delta=(g['pos'][j]-g['pos'][r])/g['radius']
        if not self.use_position:delta=torch.zeros_like(delta)
        for _ in range(2):
            message=self.message(torch.cat((x[r],x[j],delta,delta.norm(dim=1,keepdim=True)),-1))
            mean=add(message,r,len(x))/add(x.new_ones(len(r)),r,len(x)).clamp_min(1)[:,None]
            x=(x+.1*self.update(torch.cat((self.norm(x),mean),-1)))*active[:,None]
        return (x+.1*self.ff(self.norm(x)))*active[:,None]

class FixedCore:
    def configure_core(self,task,arm):
        self.dataset_task=task;self.variant=arm
        self.use_position=arm not in ('no_position','no_position_scale')
        self.use_scale=arm not in ('no_scale','no_position_scale')
        with torch.random.fork_rng(devices=[]):processor=Processor()
        processor.load_state_dict(self.processor.state_dict());processor.use_position=self.use_position
        processor.use_context=arm!='no_context';self.processor=processor

    def embedding(self,g):
        graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass']
        if self.variant=='shuffle_regions':
            # Permute within-cell content only, preserving the receiving region's shared code.
            xyz=g['pos']-g['pos'].mean(0)
            code=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))
            scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
            code=code+self.scale(scale)
            matched=g['has_g']&g['has_p'];perm=g['surface_content_permutation'];point=point.clone()
            point[matched]=(point-code)[perm[matched]]+code[matched]
        mass=gm/gm.sum().clamp_min(1e-12)+pm/pm.sum().clamp_min(1e-12)
        active=g['has_g']|g['has_p'];count=(g['has_g'].to(graph.dtype)+g['has_p'].to(graph.dtype)).clamp_min(1)
        base=(graph+point)/count[:,None];matched=g['has_g']&g['has_p'];delta=torch.zeros_like(base)
        if matched.any():delta[matched]=self.joint(torch.cat((graph[matched],point[matched]),-1))
        x=self.processor(base+.1*delta,g,mass,active)
        return self.readout(self.pool(x,g,mass,active))

    def forward(self,g):
        y=self.head(self.embedding(g))
        return y.sigmoid().squeeze(-1) if self.dataset_task=='h01_global' else y

class H01Model(FixedCore,H01Base):
    def __init__(self,task,arm):
        super().__init__('local' if task=='h01_local' else 'global','full');self.configure_core(task,arm)
class FlyModel(FixedCore,FlyBase):
    def __init__(self,task,arm):
        super().__init__(6,'full');self.configure_core(task,arm)
def Model(task,arm):return FlyModel(task,arm) if task=='flywire_coarse' else H01Model(task,arm)

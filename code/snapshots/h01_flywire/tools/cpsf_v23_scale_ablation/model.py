"""Explicit encoding ablations; physical support and task query stay unchanged."""
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from tools.cpsf_v21_segment_units.model import add, rotary
from tools.cpsf_v22_shared_region.model import Processor as BaseProcessor, D
from tools.cpsf_v23_residual_region.model import Model as BaseModel

ARMS=('full','no_position','no_scale','no_position_scale','thumbnail','uniform_budget')
ENCODING_ARMS=('no_position','no_scale','no_position_scale')
RESOLUTION_ARMS=('thumbnail','uniform_budget')
JOBS=[(task,arm) for arm in ENCODING_ARMS for task in ('local','global')]+[('global',arm) for arm in RESOLUTION_ARMS]

class Processor(BaseProcessor):
    use_position=True
    def forward(self,x,g,mass,active):
        # Missing modalities contribute no tokens or messages to this stream.
        x=x*active[:,None];e=g['adjacency'];e=e[active[e[:,0]]&active[e[:,1]]]
        a,b=e.unbind(1);r=torch.cat((a,b));j=torch.cat((b,a))
        delta=(g['pos'][j]-g['pos'][r])/g['radius']
        if not self.use_position:delta=torch.zeros_like(delta)
        for _ in range(2):
            message=self.message(torch.cat((x[r],x[j],delta,delta.norm(dim=1,keepdim=True)),-1))
            mean=add(message,r,len(x))/add(x.new_ones(len(r)),r,len(x)).clamp_min(1)[:,None]
            x=(x+.1*self.update(torch.cat((self.norm(x),mean),-1)))*active[:,None]
        owner=g['coarse_owner'];n=g['ncoarse'];w=mass*active;den=add(w,owner,n)
        coarse=add(x*w[:,None],owner,n)/den.clamp_min(1e-12)[:,None]
        xyz=add(g['pos']*w[:,None],owner,n)/den.clamp_min(1e-12)[:,None]
        valid=den>0
        if valid.any():
            tokens=coarse[valid];coords=xyz[valid];coords=(coords-coords.mean(0))/8.
            q,k,v=self.qkv(self.cnorm(tokens)).chunk(3,-1)
            def heads(t):return t.reshape(1,-1,4,D//4).transpose(1,2)
            q,k,v=heads(q),heads(k),heads(v)
            if self.use_position:q,k=rotary(q,coords[None]),rotary(k,coords[None])
            with sdpa_kernel(SDPBackend.MATH):z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
            z=z.transpose(1,2).reshape(-1,D);context=torch.zeros_like(coarse)
            context[valid]=self.context(z)
            x=(x+.1*context[owner])*active[:,None]
        return (x+.1*self.ff(self.norm(x)))*active[:,None]

class Model(BaseModel):
    def __init__(self,task,arm):
        assert arm in ARMS
        assert task=='global' or arm not in RESOLUTION_ARMS
        super().__init__(task,'full')
        self.ablation=arm
        self.use_position=arm not in ('no_position','no_position_scale')
        self.use_scale=arm not in ('no_scale','no_position_scale')
        # Preserve the inherited random stream, parameter order, and initial state.
        with torch.random.fork_rng(devices=[]):
            processor=Processor()
        processor.load_state_dict(self.processor.state_dict())
        processor.use_position=self.use_position
        self.processor=processor

    def stems(self,g):
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
        surface=add((hp[pj]+self.relation(g['phi']))*wp[:,None],pr,u)/g['surface_mass'].clamp_min(1e-12)[:,None]
        # Bounded local position; true physical length and resolution stay explicit.
        xyz=g['pos'];xyz=xyz-xyz.mean(0)
        position=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius'])) if self.use_position else torch.zeros_like(graph)
        scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
        code=position+(self.scale(scale) if self.use_scale else torch.zeros_like(graph))
        return (graph+code)*g['has_g'][:,None],(surface+code)*g['has_p'][:,None]


"""Reference-centered physical conditioning with identity initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from tools.cpsf_v21_segment_units.model import add, rotary
from tools.cpsf_v22_shared_region.model import Processor as BaseProcessor, D
from tools.cpsf_v23_scale_ablation.model import Model as BaseModel
ARMS=('full','no_position','no_scale','no_position_scale','hierarchy_only')
JOBS=[(task,arm) for arm in ARMS for task in ('local','global')]

class Processor(BaseProcessor):
    use_position=True
    def forward(self,x,g,mass,active):
        # Missing modalities contribute no tokens or messages to this stream.
        x=x*active[:,None];e=g['adjacency'];e=e[active[e[:,0]]&active[e[:,1]]]
        a,b=e.unbind(1);r=torch.cat((a,b));j=torch.cat((b,a))
        delta=(g['pos'][j]-g['pos'][r])/g['radius']
        if not self.use_position:delta=torch.zeros_like(delta)
        elif not g.get('legacy_geometry',False):delta=delta*g['position_gain']
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
            if self.use_position:
                phase=coords[None] if g.get('legacy_geometry',False) else coords[None]*g['position_gain']
                q,k=rotary(q,phase),rotary(k,phase)
            with sdpa_kernel(SDPBackend.MATH):z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
            z=z.transpose(1,2).reshape(-1,D);context=torch.zeros_like(coarse)
            context[valid]=self.context(z)
            x=(x+.1*context[owner])*active[:,None]
        return (x+.1*self.ff(self.norm(x)))*active[:,None]

class Model(BaseModel):
    def __init__(self,task,arm):
        assert arm in ARMS
        super().__init__(task,'full' if arm=='hierarchy_only' else arm)
        self.variant=arm
        self.position_alpha=nn.Parameter(torch.zeros(()))
        self.scale_alpha=nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng(devices=[]):processor=Processor()
        processor.load_state_dict(self.processor.state_dict());processor.use_position=self.use_position
        self.processor=processor

    def gains(self):
        if self.variant=='hierarchy_only':return self.position_alpha.new_ones(()),self.scale_alpha.new_ones(())
        gp=torch.tanh(self.position_alpha) if self.use_position else self.position_alpha.new_zeros(())
        gs=torch.tanh(self.scale_alpha) if self.use_scale else self.scale_alpha.new_zeros(())
        return gp,gs

    def forward(self,g):
        gp,gs=self.gains()
        return super().forward(dict(g,position_gain=gp,legacy_geometry=self.variant=='hierarchy_only'))

    def stems(self,g):
        if self.variant=='hierarchy_only':return super().stems(g)
        gp,gs=self.gains()
        u=g['nregions'];hg=self.graph(g['sf']);px=self.point(g['pf'])
        den=g['patch_mass'].clamp_min(1e-12)[:,None]
        mu=add(px*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        second=add(px.square()*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        hp=self.patch(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))
        sr,sn=g['sr'],g['sn'];pr,pj=g['pr'],g['pj']
        path=(g['s'][sn]-g['pos'][sr])/g['radius']
        path_code=gp*(self.path_position(path)-self.path_position(torch.zeros_like(path))) if self.use_position else torch.zeros_like(hg[sn])
        graph=add((hg[sn]+path_code)*g['sw'][:,None],sr,u)/g['graph_mass'].clamp_min(1e-12)[:,None]
        wp=g['pw']*g['patch_mass'][pj]
        surface=add((hp[pj]+self.relation(g['phi']))*wp[:,None],pr,u)/g['surface_mass'].clamp_min(1e-12)[:,None]
        # Bounded local position; true physical length and resolution stay explicit.
        xyz=g['pos']
        if 'query' in g:reference=g['query']
        else:
            gm=g['graph_mass'];pm=g['surface_mass'];mass=gm/gm.sum().clamp_min(1e-12)+pm/pm.sum().clamp_min(1e-12)
            reference=(xyz*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-12)
        xyz=xyz-reference
        position=gp*(self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))-self.position(torch.zeros_like(xyz))) if self.use_position else torch.zeros_like(graph)
        scale=torch.stack(((g['length']/g['radius']).log1p(),torch.full_like(g['length'],g['radius']/2.).log(),torch.full_like(g['length'],g.get('voxel',.5)/.5).log()),1)
        code=position+(gs*(self.scale(scale)-self.scale(torch.zeros_like(scale))) if self.use_scale else torch.zeros_like(graph))
        return (graph+code)*g['has_g'][:,None],(surface+code)*g['has_p'][:,None]

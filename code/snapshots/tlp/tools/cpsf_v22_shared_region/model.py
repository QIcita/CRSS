"""One common region stream, sparse local propagation, coarse XYZ-RoPE context."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from tools.cpsf_v21_segment_units.model import mlp,add,rotary
D=96
ARMS=('full','no_fusion','point','skeleton')

class Processor(nn.Module):
    def __init__(self):
        super().__init__();self.message=mlp(2*D+4,D);self.update=mlp(2*D,D)
        self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.context=nn.Linear(D,D)
        self.cnorm=nn.LayerNorm(D);self.ff=mlp(D,D)

    def forward(self,x,g,mass,active):
        # Missing modalities contribute no tokens or messages to this stream.
        x=x*active[:,None];e=g['adjacency'];e=e[active[e[:,0]]&active[e[:,1]]]
        a,b=e.unbind(1);r=torch.cat((a,b));j=torch.cat((b,a))
        delta=(g['pos'][j]-g['pos'][r])/g['radius']
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
            q=rotary(heads(q),coords[None]);k=rotary(heads(k),coords[None]);v=heads(v)
            with sdpa_kernel(SDPBackend.MATH):z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
            z=z.transpose(1,2).reshape(-1,D);context=torch.zeros_like(coarse)
            context[valid]=self.context(z)
            x=(x+.1*context[owner])*active[:,None]
        return (x+.1*self.ff(self.norm(x)))*active[:,None]

class Model(nn.Module):
    def __init__(self,task,arm):
        super().__init__();assert task in ('local','global') and arm in ARMS;self.task=task;self.arm=arm
        self.graph=mlp(4,D);self.path_position=mlp(3,D);self.point=mlp(4,D);self.patch=mlp(2*D,D)
        self.relation=mlp(5,D);self.position=mlp(3,D);self.scale=mlp(3,D)
        self.joint=mlp(2*D,D);self.processor=Processor();self.readout=mlp(D,128)
        self.head=nn.Linear(128,3 if task=='local' else 1)

    def stems(self,g):
        u=g['nregions'];hg=self.graph(g['sf']);px=self.point(g['pf'])
        den=g['patch_mass'].clamp_min(1e-12)[:,None]
        mu=add(px*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        second=add(px.square()*g['point_weight'][:,None],g['point_owner'],g['patches'])/den
        hp=self.patch(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))
        sr,sn=g['sr'],g['sn'];pr,pj=g['pr'],g['pj']
        path=(g['s'][sn]-g['pos'][sr])/g['radius']
        graph=add((hg[sn]+self.path_position(path))*g['sw'][:,None],sr,u)/g['graph_mass'].clamp_min(1e-12)[:,None]
        wp=g['pw']*g['patch_mass'][pj]
        surface=add((hp[pj]+self.relation(g['phi']))*wp[:,None],pr,u)/g['surface_mass'].clamp_min(1e-12)[:,None]
        # Bounded local position; true physical length and resolution stay explicit.
        xyz=g['pos'];xyz=xyz-xyz.mean(0)
        position=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))
        scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
        code=position+self.scale(scale)
        return (graph+code)*g['has_g'][:,None],(surface+code)*g['has_p'][:,None]

    def pool(self,x,g,mass,active):
        if self.task=='local':
            distance=(g['pos']-g['query']).norm(dim=1)
            chosen=(distance<=4.)&active
            if not chosen.any():
                assert active.any();masked=distance.masked_fill(~active,torch.inf);chosen=masked==masked.min()
            w=torch.exp(-distance[chosen]/g['radius'])*mass[chosen]
            return (x[chosen]*w[:,None]).sum(0)/w.sum().clamp_min(1e-12)
        return (x*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-12)

    def forward(self,g):
        graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass']
        # Normalize measures separately; arclength and empirical point mass are
        # never added as if they had the same physical unit.
        wg=gm/gm.sum().clamp_min(1e-12);wp=pm/pm.sum().clamp_min(1e-12)
        if self.arm=='full':
            active=g['has_g']|g['has_p'];mass=wg+wp
            x=self.joint(torch.cat((graph,point),-1))
            x=self.processor(x,g,mass,active);z=self.pool(x,g,mass,active)
        else:
            streams=[]
            for x,mass,active in ((graph,wg,g['has_g']),(point,wp,g['has_p'])):
                if active.any():streams.append(self.pool(self.processor(x,g,mass,active),g,mass,active))
            assert streams;z=torch.stack(streams).mean(0)
        y=self.head(self.readout(z))
        return y if self.task=='local' else y.sigmoid().squeeze(-1)

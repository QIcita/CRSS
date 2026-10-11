"""One local integral, no reverse fusion, no QSCR bypass, no alignment loss."""
import torch
from torch import nn
from .geometry import pack,point_view,skeleton_view

def add(x,i,n):return x.new_zeros((n,*x.shape[1:])).index_add(0,i,x)
def mlp(a,b):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,b))

class Surface(nn.Module):
    def __init__(self,d):
        super().__init__();self.stem=mlp(4,d);self.output=mlp(2*d,d)
    def forward(self,g):
        x=self.stem(g['pf']);w=g['w'][:,None];den=g['mass'][:,None].clamp_min(1e-12)
        mu=add(x*w,g['owner'],g['m'])/den
        second=add(x.square()*w,g['owner'],g['m'])/den
        return self.output(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))

class Integral(nn.Module):
    def __init__(self,d):
        super().__init__();self.kernel=nn.Sequential(nn.Linear(d+12,32),nn.GELU(),nn.Linear(32,1))
        self.null=nn.Linear(d,1);self.value=mlp(d+12,d)
    def forward(self,gstate,surface,mass,rel,mode='measure'):
        r,j=rel['receiver'],rel['sender'];n=len(gstate)
        if not len(r):return gstate.new_zeros(gstate.shape),gstate.new_ones(n)
        phi=rel['phi'];donor=rel['donor'] if mode=='wrong' else j
        mu=torch.ones_like(mass[j]) if mode=='count' else mass[j]
        rho=mu/add(mu,r,n)[r].clamp_min(1e-12)
        logits=self.kernel(torch.cat((gstate[r],phi),-1)).squeeze(-1)+rho.clamp_min(1e-30).log()
        null=self.null(gstate).squeeze(-1)
        maximum=logits.new_full((n,),-torch.inf).scatter_reduce(0,r,logits.detach(),'amax')
        maximum=torch.maximum(maximum,null.detach());ex=torch.exp(logits-maximum[r]);ne=torch.exp(null-maximum)
        denom=add(ex,r,n)+ne;a=ex/denom[r]
        u=add(a[:,None]*self.value(torch.cat((surface[donor],phi),-1)),r,n)
        return u,ne/denom

class Encoder(nn.Module):
    def __init__(self,d=64):
        super().__init__();self.d=d;self.surface=Surface(d);self.graph_stem=mlp(4,d)
        self.message=mlp(2*d+4,d);self.graph_update=mlp(2*d,d);self.graph_norm=nn.LayerNorm(d)
        self.integral=Integral(d);self.combine=mlp(2*d,d)
    def forward(self,g,arm):
        p=self.surface(g);h=self.graph_stem(g['sf'])
        if arm in ('point','geometry'):h=torch.zeros_like(h)
        else:
            a,b=g['e'].unbind(1);r=torch.cat((a,b));j=torch.cat((b,a));delta=g['s'][j]-g['s'][r]
            for _ in range(2):
                msg=self.message(torch.cat((h[r],h[j]-h[r],delta,delta.norm(dim=-1,keepdim=True)),-1))
                den=add(h.new_ones(len(r)),r,len(h)).clamp_min(1)[:,None]
                agg=add(msg,r,len(h))/den;h=self.graph_norm(h+self.graph_update(torch.cat((h,agg),-1)))
        rel=g
        if arm=='geometry':
            rel=dict(g);rel['phi']=g['phi'].clone();rel['phi'][:,10]=0
        if arm=='point':
            h=p.new_zeros((g.get('query_n',0),self.d));rel=g.get('query',dict(receiver=g['receiver'],sender=g['sender'],donor=g['donor'],phi=g['phi']))
        u,null=self.integral(h,p,g['mass'],rel,arm)
        return self.combine(torch.cat((h,u),-1)),p,null

class Model(nn.Module):
    def __init__(self,task,arm,d=64):
        super().__init__();self.task=task;self.arm=arm;self.encoder=Encoder(d)
        self.head=nn.Sequential(mlp(d if task=='h01' else 3*d,d),nn.Linear(d,3 if task=='h01' else 2))
    def encode(self,gs,queries=None):
        if self.arm=='point':
            assert queries is not None
            gs=[point_view(g,g['s'][q].numpy().reshape(1,3)) for g,q in zip(gs,queries)]
        elif self.arm=='skeleton':gs=[skeleton_view(g) for g in gs]
        packed,offsets=pack(gs,next(self.parameters()).device)
        h,p,null=self.encoder(packed,self.arm);ns,ms=offsets
        if self.arm=='point':ns=packed['query_offsets']
        return [(h[a:b],p[c:d],packed['mass'][c:d],null[a:b]) for a,b,c,d in zip(ns[:-1],ns[1:],ms[:-1],ms[1:])]
    def forward(self,gs,queries=None):
        zs=self.encode(gs,queries);features=[]
        for i,(h,p,mass,null) in enumerate(zs):
            if self.task=='h01':features.append(h[0 if self.arm=='point' else queries[i]])
            else:
                sf=(p*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-12)
                features.append(torch.cat((h[0],h.mean(0),sf)))
        logits=self.head(torch.stack(features))
        return logits if self.task=='h01' else logits-logits.mean(1,keepdim=True)

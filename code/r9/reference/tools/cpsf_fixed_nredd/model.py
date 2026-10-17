import torch
from torch import nn
from tools.cpsf_fixed_core.model import H01Model

class Model(H01Model):
    """Identical region operators; only task readouts and supervised heads change."""
    def __init__(self,task,arm):
        super().__init__('h01_local',arm)
        self.nredd_task=task;self.arm=arm
        self.head=nn.Linear(128,1)
        if task=='tlp':self.head=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,1))

    def streams(self,g):
        graph,point=self.stems(g);gm=g['graph_mass'];pm=g['surface_mass']
        wg=gm/gm.sum().clamp_min(1e-12);wp=pm/pm.sum().clamp_min(1e-12)
        if self.arm=='shuffle_regions':
            xyz=g['pos']-g['pos'].mean(0)
            code=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))
            scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),torch.full_like(g['length'],.5).log1p()),1)
            code=code+self.scale(scale);mask=g['has_g']&g['has_p'];perm=g['surface_content_permutation']
            point=point.clone();point[mask]=(point-code)[perm[mask]]+code[mask]
        if self.arm=='no_fusion':
            return [(self.processor(x,g,w,m),w,m) for x,w,m in ((graph,wg,g['has_g']),(point,wp,g['has_p'])) if m.any()]
        active=g['has_g']|g['has_p'];mass=wg+wp
        count=(g['has_g'].float()+g['has_p'].float()).clamp_min(1)
        x=(graph+point)/count[:,None]
        if self.arm!='mean_only':
            mask=g['has_g']&g['has_p'];delta=torch.zeros_like(x)
            if mask.any():delta[mask]=self.joint(torch.cat((graph[mask],point[mask]),-1))
            x=x+.1*delta
        return [(self.processor(x,g,mass,active),mass,active)]

    @staticmethod
    def query_pool(x,g,mass,active):
        # Query positions select readout only, including the surface-only control.
        d=torch.cdist(g['query_xyz'],g['pos']);selected=(d<=4)&active[None]
        missing=~selected.any(1)
        nearest=d.masked_fill(~active[None],torch.inf).argmin(1)
        selected=selected.clone();selected[missing,nearest[missing]]=True
        weight=torch.exp(-d/g['radius'])*mass[None]*selected
        # Stable nearest fallback for queries far outside retained surfaces.
        den=weight.sum(1,keepdim=True);empty=den[:,0]<1e-30
        weight[empty]=0;weight[empty,nearest[empty]]=1
        return weight/weight.sum(1,keepdim=True).clamp_min(1e-30)@x

    def forward(self,g):
        streams=self.streams(g)
        q=torch.stack([self.query_pool(x,g,w,m) for x,w,m in streams]).mean(0)
        if self.nredd_task=='ted':
            score=self.head(self.readout(q)).reshape(1)
            return torch.stack((-score,score),-1)[0]
        whole=torch.stack([(x*w[:,None]).sum(0)/w.sum().clamp_min(1e-12) for x,w,m in streams]).mean(0)
        q=self.readout(q);whole=self.readout(whole);n=len(q);parts=[]
        for start in range(0,n,32):
            a=q[start:start+32,None];b=q[None]
            features=torch.cat((a+b,(a-b).abs(),whole.expand(len(a),n,-1)),-1)
            parts.append(self.head(features).squeeze(-1))
        return torch.cat(parts)

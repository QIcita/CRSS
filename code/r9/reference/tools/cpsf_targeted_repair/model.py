"""Two separate one-factor interventions; inherited frozen shared cores."""
import numpy as np
import torch
from torch import nn
from scipy.sparse.csgraph import connected_components
from tools.cpsf_fixed_nredd.model import Model as TEDBase
from tools.cpsf_directional.model import Model as TLPBase

class TEDModel(TEDBase):
    def __init__(self,repair):super().__init__('ted','full');self.repair=repair
    def stems(self,g):
        if self.repair:g=dict(g,pf=torch.cat((g['pf_directional'][:,:3],g['pf'][:,3:]),1))
        return super().stems(g)

def pair_geometry(xyz,observed):
    """Seven symmetric channels; uses observed adjacency only, never target edges."""
    xyz=np.asarray(xyz,np.float64);obs=np.asarray(observed,bool).copy();np.fill_diagonal(obs,False)
    assert obs.shape==(len(xyz),len(xyz)) and np.array_equal(obs,obs.T)
    delta=xyz[None]-xyz[:,None];distance=np.linalg.norm(delta,axis=-1)
    direction=delta/np.maximum(distance[...,None],1e-12)
    alignment=np.zeros(distance.shape)
    for i in range(len(xyz)):
        neighbor=np.flatnonzero(obs[i]);outward=xyz[i]-xyz[neighbor]
        outward/=np.maximum(np.linalg.norm(outward,axis=1,keepdims=True),1e-12)
        if len(neighbor):alignment[i]=np.maximum(0,(direction[i]@outward.T).max(1))
    _,component=connected_components(obs,directed=False)
    result=np.concatenate((abs(delta)/20.,distance[...,None]/20.,
        np.minimum(alignment,alignment.T)[...,None],np.maximum(alignment,alignment.T)[...,None],
        (component[:,None]==component[None])[...,None]),-1).astype(np.float32)
    assert np.isfinite(result).all() and np.array_equal(result,result.transpose(1,0,2))
    return result

class TLPModel(TLPBase):
    def __init__(self,repair):
        super().__init__('tlp','full','directional_xyz');self.repair=repair
        if repair:
            old=self.head[0]
            # Preserve original weights and RNG; zero new columns give exact initial output parity.
            with torch.random.fork_rng(devices=[]):new=nn.Linear(old.in_features+7,old.out_features)
            with torch.no_grad():
                new.weight.zero_();new.weight[:,:old.in_features].copy_(old.weight);new.bias.copy_(old.bias)
            self.head[0]=new
    def forward(self,g):
        if not self.repair:return super().forward(g)
        streams=self.streams(g)
        q=torch.stack([self.query_pool(x,g,w,m) for x,w,m in streams]).mean(0)
        contexts=[]
        for x,w,m in streams:
            whole=(x*w[:,None]).sum(0)/w.sum().clamp_min(1e-12)
            bridge,valid=self.corridor_pool(x,g,w,m)
            context=torch.where(valid[:,None],.5*(whole[None]+bridge),whole[None])
            contexts.append(torch.cat((whole[None],context),0))
        context=self.readout(torch.stack(contexts).mean(0));q=self.readout(q);n=len(q);parts=[]
        for start in range(0,n,32):
            a=q[start:start+32,None];b=q[None];c=context[g['pair_lookup'][start:start+32]]
            features=torch.cat((a+b,(a-b).abs(),c,g['pair_geometry'][start:start+32]),-1)
            parts.append(self.head(features).squeeze(-1))
        return torch.cat(parts)
def Model(task,arm):return (TEDModel if task=='ted' else TLPModel)(arm=='repair')

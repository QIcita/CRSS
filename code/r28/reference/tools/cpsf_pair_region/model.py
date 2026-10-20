import torch
from tools.cpsf_fixed_nredd.model import Model as Base

class Model(Base):
    def __init__(self,task,arm):
        super().__init__(task,'full' if arm=='shuffle_corridor' else arm)
        self.pair_arm=arm
        self.use_corridor=True

    def corridor_pool(self,x,g,mass,active):
        row,col=g['bridge_row'],g['bridge_col'];p=g['pair_count']
        weight=g['bridge_weight']*mass[col]*active[col]
        den=x.new_zeros(p).index_add(0,row,weight)
        weight=weight/den[row].clamp_min(1e-12)
        incidence=torch.sparse_coo_tensor(torch.stack((row,col)),weight,(p,len(x))).coalesce()
        pooled=torch.sparse.mm(incidence,x)
        return pooled,den>0

    def forward(self,g):
        if not self.use_corridor:return super().forward(g)
        streams=self.streams(g)
        q=torch.stack([self.query_pool(x,g,w,m) for x,w,m in streams]).mean(0)
        contexts=[]
        for x,w,m in streams:
            whole=(x*w[:,None]).sum(0)/w.sum().clamp_min(1e-12)
            bridge,valid=self.corridor_pool(x,g,w,m)
            if self.pair_arm=='shuffle_corridor':bridge=bridge[g['bridge_permutation']]
            context=torch.where(valid[:,None],.5*(whole[None]+bridge),whole[None])
            # Index zero represents the diagonal, which uses original global context.
            contexts.append(torch.cat((whole[None],context),0))
        context=self.readout(torch.stack(contexts).mean(0));q=self.readout(q);n=len(q);parts=[]
        for start in range(0,n,32):
            a=q[start:start+32,None];b=q[None]
            c=context[g['pair_lookup'][start:start+32]]
            parts.append(self.head(torch.cat((a+b,(a-b).abs(),c),-1)).squeeze(-1))
        return torch.cat(parts)

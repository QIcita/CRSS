"""Frozen modality cores, with explicit local/global/regional task readouts."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ARMS,TASKS
from tools.cpsf_smm_runtime.model import make as conditioned_make
from tools.cpsf_progressive_smm.model import Model as Progressive
from tools.cpsf_smm_joint.models import add


class Model(nn.Module):
    def __init__(self,arm,task,initial):
        super().__init__()
        assert arm in ARMS and task in TASKS
        original=conditioned_make('surface_conditioned')
        if arm=='progressive_gated':
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(42);self.core=Progressive(original,True).cuda()
        else:self.core=original
        self.core.head=nn.Identity()
        self.arm=arm;self.task=task
        # Singleton microbatches retain the baseline's stateless task heads.
        # Both arms receive exactly the same head initialization per task.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            if task=='microns':
                self.head=nn.Sequential(nn.Linear(640,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,1)).cuda()
                with torch.no_grad():
                    self.head[-1].weight.mul_(.01);self.head[-1].bias.fill_(math.log(initial))
            else:
                self.head=nn.Linear(896 if task=='h01_local' else 640,3 if task=='h01_local' else 1).cuda()
                if task=='h01_global':
                    nn.init.normal_(self.head.weight,std=.001)
                    nn.init.constant_(self.head.bias,math.log(initial/(1-initial)))

    def conditioned_regions(self,query,patches,g):
        rcm=self.core.rcm;n=g['nregions'];pr,pj=g['pr'],g['pj']
        key=rcm.key(patches);value=rcm.value(patches)
        prior=g['pw']*g['patch_mass'][pj]/g['surface_mass'][pr]
        scores=[]
        for lo in range(0,len(pr),16384):
            rr,pp=pr[lo:lo+16384],pj[lo:lo+16384];phi=g['phi'][lo:lo+16384]
            def score(q,k,geo,rr=rr,pp=pp):
                return (q[rr]*k[pp]).sum(-1)/4+rcm.score_geometry(geo).squeeze(-1)
            scores.append(checkpoint(score,query,key,phi,use_reentrant=False) if torch.is_grad_enabled() else score(query,key,phi))
        log_weight=prior.log()+torch.cat(scores);maximum=log_weight.new_full((n,),-torch.inf)
        maximum.scatter_reduce_(0,pr,log_weight.detach(),reduce='amax',include_self=True)
        weights=(log_weight-maximum[pr]).exp();alpha=weights/add(weights,pr,n)[pr]
        regional=patches.new_zeros((n,96))
        for lo in range(0,len(pr),16384):
            rr,pp=pr[lo:lo+16384],pj[lo:lo+16384];phi=g['phi'][lo:lo+16384]
            def gather(v,p,a,geo,rr=rr,pp=pp):
                return add(p[:,None]*v[pp]+a[:,None]*rcm.relation(geo),rr,n)
            args=(value,alpha[lo:lo+16384],prior[lo:lo+16384],phi)
            regional=regional+(checkpoint(gather,*args,use_reentrant=False) if torch.is_grad_enabled() else gather(*args))
        return rcm.output(regional)

    def features(self,x):
        raw=next(self.core.structures([x]));patches=self.core.cmm(x['g'])
        if self.arm=='progressive_gated':
            structure=self.core.smm.refine(raw,x)
            if self.task=='microns':
                delta=self.core.rcm(structure,patches,x,regional_output=True)
            else:delta=self.core.rcm(structure,patches,x)
        else:
            query=self.core.smm.regional_query(raw,x)
            delta=self.conditioned_regions(query,patches,x['g']) if self.task=='microns' else self.core.rcm(query,patches,x['g'],True)
        if self.task=='microns':
            branch=add(raw.branches[x['jb']]*x['jw'][:,None],x['jr'],x['g']['nregions'])
            skip=torch.cat((F.normalize(branch,dim=-1),F.normalize(raw.topology,dim=-1).expand(len(branch),-1)),dim=-1)
            return (skip+delta)[x['query_index']]
        feature=raw.global_feature+delta
        if self.task=='h01_local':
            local=raw.branches[x['batch'].offset[x['query_component']]]
            feature=torch.cat((feature,F.normalize(local,dim=-1)))
        return feature

    def forward(self,x):
        z=self.head(self.features(x))
        if self.task=='microns':return z.squeeze(-1).clamp(-12,8).exp()*x['length']
        if self.task=='h01_global':return z[0].sigmoid()
        return z


def make(arm,task,initial):
    model=Model(arm,task,initial)
    if arm=='progressive_gated':
        # Local copy differs only by exposing the existing pre-pooling state.
        # The frozen FlyWire module and its running workers remain untouched.
        from .regional_core import Coupling
        model.core.rcm.__class__=Coupling
    return model.cuda()

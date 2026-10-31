"""Intra-modal interaction, support-aligned coupling, and cross-scale residuals."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ARMS
from tools.cpsf_smm_runtime.model import RuntimeModel, make as parent_make
from tools.cpsf_smm_joint.models import SkeletonMorphology, add


def normalize(x):
    return F.layer_norm(x,(x.shape[-1],))


class PairResidual(nn.Module):
    """Shared rank-16 interaction factors, two independently bounded outputs."""
    def __init__(self,left,right,rank=16):
        super().__init__()
        self.left=nn.Linear(left,rank,bias=False)
        self.right=nn.Linear(right,rank,bias=False)
        self.to_left=nn.Parameter(torch.zeros(left,rank))
        self.to_right=nn.Parameter(torch.zeros(right,rank))
        self.rank=rank

    def product(self,left,right):
        return torch.tanh(self.left(normalize(left)))*torch.tanh(self.right(normalize(right)))/math.sqrt(self.rank)

    @staticmethod
    def project(product,weight,reference):
        bounded=weight/torch.sqrt(1+weight.square().sum())
        scale=reference.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        return .1*scale*F.linear(product,bounded)

    def forward(self,left,right):
        product=self.product(left,right)
        return (self.project(product,self.to_left,left),self.project(product,self.to_right,right))


class CrossResidual(nn.Module):
    def __init__(self,width=96,rank=16):
        super().__init__()
        self.local=nn.Linear(width,rank,bias=False)
        self.global_context=nn.Linear(width,rank,bias=False)
        self.output=nn.Parameter(torch.zeros(width,rank))
        self.rank=rank

    def forward(self,local,global_context):
        product=torch.tanh(self.local(normalize(local)))*torch.tanh(self.global_context(normalize(global_context)))
        return PairResidual.project(product/math.sqrt(self.rank),self.output,local)


@dataclass(frozen=True)
class Structure:
    branches: torch.Tensor
    topology: torch.Tensor
    global_feature: torch.Tensor
    skip: torch.Tensor


class Skeleton(nn.Module):
    tree_states=SkeletonMorphology.tree_states

    def __init__(self,original):
        super().__init__()
        self.tree=original.tree
        self.persistence=original.persistence
        self.interaction=PairResidual(256,384)

    def refine(self,raw,x):
        roots=raw.branches[x['batch'].offset.long()]
        root=(roots*x['mass'][:,None]).sum(0)/x['mass'].sum()
        pair=self.interaction
        branches=raw.branches+pair.project(pair.product(raw.branches,raw.topology),pair.to_left,raw.branches)
        topology=raw.topology+pair.project(pair.product(root,raw.topology),pair.to_right,raw.topology)
        roots=branches[x['batch'].offset.long()]
        root=(roots*x['mass'][:,None]).sum(0)/x['mass'].sum()
        whole=torch.cat((F.normalize(root,dim=-1),F.normalize(topology,dim=-1)))
        return Structure(branches,topology,whole,raw.global_feature)


class Coupling(nn.Module):
    def __init__(self,original,position,scale,gated):
        super().__init__()
        self.position=position;self.scale=scale
        self.key=original.key;self.value=original.value
        self.relation=original.relation;self.score_geometry=original.score_geometry
        self.output=original.output
        self.region_projection=nn.Linear(256,96,bias=False)
        self.global_projection=nn.Linear(640,96,bias=False)
        self.query=nn.Linear(96,16)
        self.geometry_query=nn.Linear(96,16,bias=False)
        # Independent implementation of the query-dependent sigmoid output gate
        # from Qiu et al., NeurIPS 2025; not a copied Qwen attention backbone.
        self.content_gate=nn.Linear(96,96,bias=False)
        nn.init.zeros_(self.content_gate.weight)
        self.local_pair=PairResidual(96,96)
        self.global_pair=PairResidual(96,96)
        self.cross_skeleton=CrossResidual()
        self.cross_surface=CrossResidual()
        self.gated=gated

    def forward(self,structure,patches,x,trace=False):
        g=x['g'];n=g['nregions'];rr_all,pp_all=g['pr'],g['pj']
        skeleton_mask=g['has_g'].bool();surface_mask=g['surface_mass']>0
        matched=skeleton_mask&surface_mask
        branch=add(structure.branches[x['jb']]*x['jw'][:,None],x['jr'],n)
        skeleton=self.region_projection(normalize(branch))*skeleton_mask[:,None]
        skeleton_global=self.global_projection(structure.global_feature)
        surface_global=self.value((patches*g['patch_mass'][:,None]).sum(0)/g['patch_mass'].sum())
        xyz=g['pos']-g['pos'].mean(0)
        scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),
                           torch.full_like(g['length'],.5).log1p()),1)
        geometry=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))+self.scale(scale)
        query=self.query(normalize(skeleton))+self.geometry_query(normalize(geometry))
        keys=self.key(patches);values=self.value(patches)
        prior=g['pw']*g['patch_mass'][pp_all]/g['surface_mass'][rr_all]
        scores=[]
        for lo in range(0,len(rr_all),16384):
            rr,pp=rr_all[lo:lo+16384],pp_all[lo:lo+16384];phi=g['phi'][lo:lo+16384]
            def score(q,k,phi,rr=rr,pp=pp):
                return (q[rr]*k[pp]).sum(-1)/4+self.score_geometry(phi).squeeze(-1)
            scores.append(checkpoint(score,query,keys,phi,use_reentrant=False) if torch.is_grad_enabled() else score(query,keys,phi))
        log_weight=prior.log()+torch.cat(scores)
        maximum=log_weight.new_full((n,),-torch.inf)
        maximum.scatter_reduce_(0,rr_all,log_weight.detach(),reduce='amax',include_self=True)
        weights=(log_weight-maximum[rr_all]).exp()
        alpha=weights/add(weights,rr_all,n)[rr_all]
        alpha=torch.where(matched[rr_all],alpha,prior)
        content=patches.new_zeros((n,96));relation=patches.new_zeros((n,96))
        for lo in range(0,len(rr_all),16384):
            rr,pp=rr_all[lo:lo+16384],pp_all[lo:lo+16384];phi=g['phi'][lo:lo+16384]
            prob,a=alpha[lo:lo+16384],prior[lo:lo+16384]
            def gather(value,prob,a,phi,rr=rr,pp=pp):
                return add(prob[:,None]*value[pp],rr,n),add(a[:,None]*self.relation(phi),rr,n)
            c,b=checkpoint(gather,values,prob,a,phi,use_reentrant=False) if torch.is_grad_enabled() else gather(values,prob,a,phi)
            content=content+c;relation=relation+b
        gate_logits=self.content_gate(normalize(skeleton))
        gate=torch.sigmoid(gate_logits) if self.gated else torch.ones_like(gate_logits)+gate_logits*0.
        gate=torch.where(matched[:,None],gate,torch.ones_like(gate))
        surface=gate*content+relation
        ds,dp=self.local_pair(skeleton,surface)
        ds=ds*matched[:,None];dp=dp*matched[:,None]
        skeleton_local=skeleton+ds;surface_local=surface+dp
        # Cross-scale inputs retain their modality of origin. The globally
        # coupled residual is read out separately, avoiding a feedback loop.
        dsg,dpg=self.global_pair(skeleton_global,surface_global)
        cs=self.cross_skeleton(skeleton_local,surface_global)*skeleton_mask[:,None]
        cp=self.cross_surface(surface_local,skeleton_global)*surface_mask[:,None]
        skeleton_final=skeleton_local+cs;surface_final=surface_local+cp
        regional=surface_final+(skeleton_final-skeleton)
        mass=g['surface_mass']/g['surface_mass'].sum()
        pooled=(regional*mass[:,None]).sum(0)+dsg+dpg
        delta=self.output(pooled)
        if trace:
            return delta,dict(regional=regional,pooled=pooled,alpha=alpha,prior=prior,gate=gate,matched=matched,
                              skeleton=skeleton,skeleton_final=skeleton_final,content=content,relation=relation,
                              surface=surface,surface_final=surface_final,skeleton_global=skeleton_global,surface_global=surface_global)
        return delta


class Model(nn.Module):
    structures=RuntimeModel.structures

    def __init__(self,original,gated):
        super().__init__()
        self.smm=Skeleton(original.smm)
        self.cmm=original.cmm
        self.rcm=Coupling(original.rcm,original.smm.position,original.smm.scale,gated)
        self.head=original.head
        self.checkpoint_samples=False

    def forward_group(self,items):
        features=[]
        for x,raw in zip(items,self.structures(items)):
            structure=self.smm.refine(raw,x)
            patches=self.cmm(x['g'])
            features.append(structure.skip+self.rcm(structure,patches,x))
        return self.head(torch.stack(features))

    def forward(self,x):
        return self.forward_group([x])[0]


def make(arm):
    if arm not in ARMS:raise ValueError(arm)
    original=parent_make('surface_conditioned')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        model=Model(original,gated=arm=='progressive_gated').cuda()
    return model

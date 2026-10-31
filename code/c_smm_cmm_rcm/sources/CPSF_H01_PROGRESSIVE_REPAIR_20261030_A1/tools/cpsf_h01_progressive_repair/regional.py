"""Support-aligned coupling followed by the existing H01 regional processor."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tools.cpsf_progressive_smm.model import normalize
from tools.cpsf_smm_joint.models import add

class RegionalCoupling(nn.Module):
    def __init__(self,source,processor,readout,task):
        super().__init__()
        for name,module in source.named_children():
            if name!='output':self.add_module(name,module)
        self.gated=source.gated
        self.processor=processor
        self.readout=readout
        self.task=task

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
        count=(skeleton_mask.to(patches.dtype)+surface_mask.to(patches.dtype)).clamp_min(1)
        active=skeleton_mask|surface_mask
        regional=((skeleton_final+geometry)*skeleton_mask[:,None]+
                  (surface_final+geometry)*surface_mask[:,None])/count[:,None]
        regional=regional+(dsg+dpg)*active[:,None]
        gm,pm=g['graph_mass'],g['surface_mass']
        mass=gm/gm.sum().clamp_min(1e-12)+pm/pm.sum().clamp_min(1e-12)
        processed=self.processor(regional,g,mass,active)
        if self.task=='h01_local':
            distance=(g['pos']-g['query']).norm(dim=1)
            chosen=(distance<=4.)&active
            if not chosen.any():
                masked=distance.masked_fill(~active,torch.inf);chosen=masked==masked.min()
            weight=torch.exp(-distance[chosen]/g['radius'])*mass[chosen]
            pooled=(processed[chosen]*weight[:,None]).sum(0)/weight.sum().clamp_min(1e-12)
        else:pooled=(processed*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-12)
        output=self.readout(pooled)
        if trace:return output,dict(regional=regional,processed=processed,mass=mass,active=active,pooled=pooled,gate=gate)
        return output

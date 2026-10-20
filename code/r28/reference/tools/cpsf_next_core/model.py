"""Scale-calibrated dual-measure correspondence network (experimental).
Utonia-inspired Q/K-only rotary; custom sparse geometry-conditioned operator.
Not Utonia reproduction, not causal discovery; no pretrained dependencies.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
D=96;HEADS=4;DH=D//HEADS

def add(v,ix,n):return v.new_zeros((n,*v.shape[1:])).index_add(0,ix,v)
def softmax_group(logit,receiver,n):
 peak=logit.new_full((n,logit.shape[1]),-torch.inf).scatter_reduce(0,receiver[:,None].expand_as(logit),logit.detach(),reduce='amax')
 x=(logit-peak[receiver]).exp();return x/add(x,receiver,n)[receiver].clamp_min(1e-12)
def rotate(x,xyz):
 # x (...,H,24), xyz (...,3); pair-common reference makes translation cancel.
 v=x.reshape(*x.shape[:-1],3,DH//6,2);freq=10.**(-torch.arange(DH//6,device=x.device,dtype=x.dtype)/(DH//6))
 a=xyz[...,None,:,None]*freq;co,si=a.cos(),a.sin();u,w=v[...,0],v[...,1]
 return torch.stack((u*co-w*si,u*si+w*co),-1).flatten(-3)
def mlp(a,b):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,b))

class SparseRelation(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(D);self.q=nn.Linear(D,D);self.k=nn.Linear(D,D);self.v=nn.Linear(D,D);self.relation=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,HEADS));self.out=nn.Linear(D,D)
 def forward(self,x,pos,scale,mass,edges,use_rope=True):
  r,j=edges.T;z=self.norm(x);q=self.q(z).view(-1,HEADS,DH)[r];k=self.k(z).view(-1,HEADS,DH)[j]
  h=torch.sqrt(scale[r]*scale[j]).clamp_min(1e-6);delta=pos[j]-pos[r];rel=delta/h[:,None]
  if use_rope:k=rotate(k,rel)
  phi=torch.cat((rel.clamp(-20,20),delta.norm(dim=1,keepdim=True).log1p(),h.log1p()[:,None],(scale[j]/scale[r]).log().clamp(-8,8)[:,None]),1)
  score=(q*k).sum(-1)/math.sqrt(DH)+self.relation(phi)+mass[j].clamp_min(1e-12).log()[:,None]
  a=softmax_group(score,r,len(x));v=self.v(z).view(-1,HEADS,DH)[j];update=add(a[...,None]*v,r,len(x)).flatten(1)
  return x+.1*self.out(update)

class Correspondence(nn.Module):
 def __init__(self):
  super().__init__();self.sn=nn.LayerNorm(D);self.pn=nn.LayerNorm(D);self.q=nn.Linear(D,D);self.k=nn.Linear(D,D);self.sv=nn.Linear(D,D);self.pv=nn.Linear(D,D);self.so=nn.Linear(D,D);self.po=nn.Linear(D,D);self.bias=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,HEADS))
 def forward(self,s,p,g,use_rope=True):
  r,j=g['cross'].T;sn,pn=self.sn(s),self.pn(p);q=self.q(sn).view(-1,HEADS,DH)[r];k=self.k(pn).view(-1,HEADS,DH)[j]
  delta=g['ppos'][j]-g['spos'][r];h=torch.sqrt(g['sscale'][r]*g['pscale'][j]).clamp_min(1e-6);rel=delta/h[:,None]
  if use_rope:k=rotate(k,rel)
  phi=torch.cat((rel.clamp(-20,20),delta.norm(dim=1,keepdim=True).log1p(),h.log1p()[:,None],(g['pscale'][j]/g['sscale'][r]).log().clamp(-8,8)[:,None]),1)
  compatibility=(q*k).sum(-1)/math.sqrt(DH)+self.bias(phi)
  # Same learned correspondence, two conditional normalizations; not Sinkhorn OT.
  a=softmax_group(compatibility+g['pmass'][j].log()[:,None],r,len(s))
  b=softmax_group(compatibility+g['smass'][r].log()[:,None],j,len(p))
  sv=self.sv(sn).view(-1,HEADS,DH)[r];pv=self.pv(pn).view(-1,HEADS,DH)[j]
  ds=add(a[...,None]*pv,r,len(s)).flatten(1);dp=add(b[...,None]*sv,j,len(p)).flatten(1)
  return s+.1*self.so(ds),p+.1*self.po(dp)

class Context(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.out=nn.Linear(D,D);self.physical=mlp(1,D)
 def forward(self,s,p,g,use_rope=True):
  n=len(g['cpos']);sm=g['smass']/add(g['smass'],g['sbatch'],g['batch_size'])[g['sbatch']];pm=g['pmass']/add(g['pmass'],g['pbatch'],g['batch_size'])[g['pbatch']]
  mass=add(sm,g['sowner'],n)+add(pm,g['powner'],n)
  x=(add(s*sm[:,None],g['sowner'],n)+add(p*pm[:,None],g['powner'],n))/mass.clamp_min(1e-12)[:,None]
  out=[];start=0
  for b,(_,_,nc,_) in enumerate(g['shapes']):
   token=x[start:start+nc];pos=g['cpos'][start:start+nc];h=g['cscale'][b];xyz=(pos-pos.mean(0))/h
   q,k,v=self.qkv(self.norm(token)).chunk(3,-1);q=q.view(nc,HEADS,DH);k=k.view(nc,HEADS,DH);v=v.view(nc,HEADS,DH)
   if use_rope:q,k=rotate(q,xyz),rotate(k,xyz)
   logit=torch.einsum('ihd,jhd->hij',q,k)/math.sqrt(DH)+mass[start:start+nc].clamp_min(1e-12).log()[None,None]
   weight=logit.softmax(-1);z=torch.einsum('hij,jhd->ihd',weight,v).flatten(1)
   out.append(self.out(z)+self.physical(h.log1p().reshape(1,1)));start+=nc
  context=torch.cat(out);return s+.1*context[g['sowner']],p+.1*context[g['powner']]

class Block(nn.Module):
 def __init__(self):
  super().__init__();self.graph=SparseRelation();self.point=SparseRelation();self.cross=Correspondence();self.context=Context();self.sn=nn.LayerNorm(D);self.pn=nn.LayerNorm(D);self.sf=mlp(D,D);self.pf=mlp(D,D)
 def forward(self,s,p,g,arm):
  s=self.graph(s,g['spos'],g['sscale'],g['smass'],g['graph_edge'],arm!='no_rope');p=self.point(p,g['ppos'],g['pscale'],g['pmass'],g['point_edge'],arm!='no_rope')
  if arm!='no_exchange':s,p=self.cross(s,p,g,arm!='no_rope')
  if arm!='no_context':s,p=self.context(s,p,g,arm!='no_rope')
  return s+.1*self.sf(self.sn(s)),p+.1*self.pf(self.pn(p))

class Model(nn.Module):
 def __init__(self,task,arm='full'):
  super().__init__();self.task=task;self.arm=arm;self.graph=mlp(8,D);self.point=mlp(4,48);self.patch=mlp(48*2+9,D);self.blocks=nn.ModuleList([Block() for _ in range(3)]);self.norm=nn.LayerNorm(D)
  self.readout=nn.Sequential(nn.Linear(4*D,192),nn.GELU(),nn.LayerNorm(192),nn.Linear(192,128),nn.GELU());self.head=nn.Linear(128,3 if task=='h01_local' else 6 if task=='flywire_coarse' else 1)
  if task=='tlp':self.pairhead=nn.Sequential(nn.Linear(128*3+7,128),nn.GELU(),nn.Linear(128,1))
 def forward(self,g):
  s=self.graph(g['sf']);local=self.point(g['pfeat']);p=self.patch(torch.cat((local.mean(1),local.max(1).values,g['pintrinsic']),-1))
  for block in self.blocks:s,p=block(s,p,g,self.arm)
  s,p=self.norm(s),self.norm(p);zs=[];si=pi=qi=0
  for ns,np_,_,nq in g['shapes']:
   ss,pp=s[si:si+ns],p[pi:pi+np_];sm,pm=g['smass'][si:si+ns],g['pmass'][pi:pi+np_]
   gs=(ss*sm[:,None]).sum(0)/sm.sum();gp=(pp*pm[:,None]).sum(0)/pm.sum()
   queries=g['query'][qi:qi+nq];sp=g['spos'][si:si+ns];pos=g['ppos'][pi:pi+np_]
   for q in queries:
    if self.task in ['h01_global','flywire_coarse']:ls,lp=ss.max(0).values,pp.max(0).values
    else:
     ds=(sp-q).norm(dim=1);dp=(pos-q).norm(dim=1);nearest=ds.argmin();h=g['sscale'][si+nearest].clamp_min(.25)
     sw=torch.exp(-ds/(2*h))*sm;pw=torch.exp(-dp/(2*h))*pm
     ls=.5*(ss[nearest]+(ss*sw[:,None]).sum(0)/sw.sum().clamp_min(1e-12));lp=(pp*pw[:,None]).sum(0)/pw.sum().clamp_min(1e-12)
    zs.append(torch.cat((ls,lp,gs,gp)))
   si+=ns;pi+=np_;qi+=nq
  z=self.readout(torch.stack(zs));y=self.head(z)
  if self.task=='ted':return torch.cat((-y,y),-1)
  if self.task=='h01_global':return y[:,0].sigmoid()
  if self.task=='tlp':
   assert g['batch_size']==1;a=z[:,None];b=z[None];context=z.mean(0).expand(len(z),len(z),-1)
   return self.pairhead(torch.cat((a+b,abs(a-b),context,g['pair_geometry']),-1)).squeeze(-1)
  return y

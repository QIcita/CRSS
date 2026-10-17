"""Common-Region Operator: observed arclength support, conditional surface
integration, one joint stream, topology-local and coarse physical context.
Custom model; uses Utonia-inspired Q/K RoPE, no external pretrained weights.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
D=96;H=4;DH=24

def mlp(a,b):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,b))
def add(v,ix,n):return v.new_zeros((n,*v.shape[1:])).index_add(0,ix,v)
def group_softmax(score,r,n):
 peak=score.new_full((n,score.shape[-1]),-torch.inf).scatter_reduce(0,r[:,None].expand_as(score),score.detach(),reduce='amax')
 a=(score-peak[r]).exp();return a/add(a,r,n)[r].clamp_min(1e-12)
def rotate(x,xyz):
 y=x.reshape(*x.shape[:-1],3,4,2);freq=10.**(-torch.arange(4,device=x.device,dtype=x.dtype)/4)
 a=xyz[...,None,:,None]*freq;u,v=y[...,0],y[...,1]
 return torch.stack((u*a.cos()-v*a.sin(),u*a.sin()+v*a.cos()),-1).flatten(-3)

class RegionalFusion(nn.Module):
 def __init__(self):
  super().__init__();self.graph=mlp(4,D);self.path=mlp(3,D);self.point=mlp(9,D);self.patch=mlp(2*D,D)
  self.gnorm=nn.LayerNorm(D);self.pnorm=nn.LayerNorm(D);self.q=nn.Linear(D,D);self.k=nn.Linear(D,D)
  self.geometry=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,H));self.surface_shape=mlp(5,D)
  self.interaction=mlp(4*D,D);self.scale=mlp(3,D)
 def forward(self,g,arm):
  n=g['nregions'];hg=self.graph(g['sf']);sr,sn=g['sr'],g['sn'];r=g['rradius']
  relative=(g['s'][sn]-g['pos'][sr])/r[sr,None]
  graph=add((hg[sn]+self.path(relative))*g['sw'][:,None],sr,n)/g['graph_mass'].clamp_min(1e-12)[:,None]
  point=self.point(g['pf']);owner=g['point_owner'];pm=g['patch_mass'].clamp_min(1e-12)[:,None]
  mu=add(point*g['point_weight'][:,None],owner,g['npatches'])/pm
  var=(add(point.square()*g['point_weight'][:,None],owner,g['npatches'])/pm-mu.square()).clamp_min(0)
  patch=self.patch(torch.cat((mu,var),-1));pr,pj=g['pr'],g['pj'];prior=(g['pw']*g['patch_mass'][pj]).clamp_min(1e-12)
  delta=(g['patch_pos'][pj]-g['pos'][pr])/r[pr,None]
  q=self.q(self.gnorm(graph)).view(-1,H,DH);k=self.k(self.pnorm(patch)).view(-1,H,DH)
  def scores(q,k,pr,pj,delta,phi,prior):
   kk=k[pj]
   if arm!='no_position':kk=rotate(kk,delta)
   return (q[pr]*kk).sum(-1)/math.sqrt(DH)+self.geometry(torch.cat((phi,delta.clamp(-20,20)),1))+prior.log()[:,None]
  # Exact incidence chunks; checkpoint recomputes activations, never truncates support.
  chunk=32768;parts=[]
  for start in range(0,len(pr),chunk):
   sl=slice(start,start+chunk)
   args=(q,k,pr[sl],pj[sl],delta[sl],g['phi'][sl],prior[sl])
   parts.append(checkpoint(scores,*args,use_reentrant=False) if torch.is_grad_enabled() else scores(*args))
  logits=torch.cat(parts)
  if arm=='uniform_incidence':logits=prior.log()[:,None].expand(-1,H)
  weights=group_softmax(logits,pr,n)
  def integrate(patch,weights,pr,pj):return add(weights[:,:,None]*patch.view(-1,H,DH)[pj],pr,n).flatten(1)
  surface=graph.new_zeros((n,D))
  for start in range(0,len(pr),chunk):
   sl=slice(start,start+chunk);args=(patch,weights[sl],pr[sl],pj[sl])
   surface=surface+(checkpoint(integrate,*args,use_reentrant=False) if torch.is_grad_enabled() else integrate(*args))
  shape=add(g['phi']*prior[:,None],pr,n)/g['surface_mass'].clamp_min(1e-12)[:,None]
  surface=(surface+self.surface_shape(shape))*g['has_p'][:,None];graph=graph*g['has_g'][:,None]
  count=(g['has_g'].float()+g['has_p'].float()).clamp_min(1);base=(graph+surface)/count[:,None]
  a,b=self.gnorm(graph),self.pnorm(surface);matched=(g['has_g']&g['has_p'])[:,None]
  if arm!='mean_only':base=base+.1*self.interaction(torch.cat((a,b,a*b,abs(a-b)),1))*matched
  if arm!='no_scale':base=base+self.scale(torch.stack((g['length'].log1p(),r.log1p(),torch.full_like(r,.5).log1p()),1))
  return base

class RegionOperator(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.geometry=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,H))
  self.edge=mlp(2*D+4,D);self.update=nn.Linear(D,D);self.cnorm=nn.LayerNorm(D);self.cqkv=nn.Linear(D,3*D);self.cout=nn.Linear(D,D);self.ff=mlp(D,D)
 def forward(self,x,g,mass,arm):
  n=len(x);e=g['adjacency'];ids=torch.arange(n,device=x.device);r=torch.cat((e[:,0],e[:,1],ids));j=torch.cat((e[:,1],e[:,0],ids))
  z=self.norm(x);q,k,v=self.qkv(z).chunk(3,-1);q=q.view(-1,H,DH)[r];k=k.view(-1,H,DH)[j]
  scale=torch.sqrt(g['rradius'][r]*g['rradius'][j]);delta=(g['pos'][j]-g['pos'][r])/scale[:,None];phi=torch.cat((delta,delta.norm(dim=1,keepdim=True)),1)
  if arm!='no_position':k=rotate(k,delta)
  a=group_softmax((q*k).sum(-1)/math.sqrt(DH)+self.geometry(phi)+mass[j].clamp_min(1e-12).log()[:,None],r,n)
  message=v[j]+self.edge(torch.cat((z[r],z[j]-z[r],phi),1));x=x+.1*self.update(add(a[:,:,None]*message.view(-1,H,DH),r,n).flatten(1))
  if arm!='no_context':
   owner=g['coarse_owner'];nc=g['ncoarse'];den=add(mass,owner,nc).clamp_min(1e-12)
   coarse=add(x*mass[:,None],owner,nc)/den[:,None];pos=add(g['pos']*mass[:,None],owner,nc)/den[:,None];outputs=[];offset=0
   for batch,(_,_,_,kcount) in enumerate(g['shapes']):
    xx=self.cnorm(coarse[offset:offset+kcount]);coords=pos[offset:offset+kcount];coords=(coords-coords.mean(0))/(2*g['radius'][batch])
    q,k,v=self.cqkv(xx).chunk(3,-1);q=q.view(-1,H,DH);k=k.view(-1,H,DH);v=v.view(-1,H,DH)
    if arm!='no_position':q,k=rotate(q,coords),rotate(k,coords)
    a=(torch.einsum('ihd,jhd->hij',q,k)/math.sqrt(DH)+den[offset:offset+kcount].log()[None,None]).softmax(-1)
    outputs.append(self.cout(torch.einsum('hij,jhd->ihd',a,v).flatten(1)));offset+=kcount
   x=x+.1*torch.cat(outputs)[owner]
  return x+.1*self.ff(self.norm(x))

class Model(nn.Module):
 def __init__(self,task,arm='full'):
  super().__init__();self.task=task;self.arm=arm;self.fusion=RegionalFusion();self.position=mlp(3,D);self.layers=nn.ModuleList([RegionOperator() for _ in range(2)])
  self.norm=nn.LayerNorm(D);self.readout=mlp(2*D,128);self.head=nn.Linear(128,3 if task=='h01_local' else 1)
 def forward(self,g):
  gm=g['graph_mass'];pm=g['surface_mass'];batch=g['rbatch'];b=g['batch_size'];wg=gm/add(gm,batch,b)[batch].clamp_min(1e-12);wp=pm/add(pm,batch,b)[batch].clamp_min(1e-12);mass=wg+wp
  x=self.fusion(g,self.arm)
  if self.arm!='no_position':
   center=add(g['pos']*mass[:,None],batch,b)/add(mass,batch,b)[:,None].clamp_min(1e-12)
   xyz=(g['pos']-center[batch])/g['rradius'][:,None]
   # Encode physical region location once in the joint stream; never give both
   # modalities a duplicated shared content offset before their association.
   x=x+self.position(torch.sign(xyz)*torch.log1p(xyz.abs()))
  for layer in self.layers:x=layer(x,g,mass,self.arm)
  x=self.norm(x);out=[];start=0
  for i,(_,_,n,_) in enumerate(g['shapes']):
   xx=x[start:start+n];w=mass[start:start+n];pos=g['pos'][start:start+n]
   if self.task=='h01_global':
    w=w/w.sum();mean=(xx*w[:,None]).sum(0);second=((xx-mean).square()*w[:,None]).sum(0);out.append(torch.cat((mean,second)))
   else:
    d=(pos-g['query'][i]).norm(dim=1);radius=g['radius'][i]
    def pool(mult):
     selected=d<=mult*radius
     if not bool(selected.any()):selected=d==d.min()
     weight=(-d[selected]/radius+w[selected].clamp_min(1e-12).log()).softmax(0)
     return (xx[selected]*weight[:,None]).sum(0)
    out.append(torch.cat((pool(1.),pool(2.))))
   start+=n
  z=self.head(self.readout(torch.stack(out)))
  if self.task=='ted':return torch.cat((-z,z),-1)
  if self.task=='h01_global':return z[:,0].sigmoid()
  return z

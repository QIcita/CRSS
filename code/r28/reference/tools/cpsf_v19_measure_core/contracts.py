"""Mechanism tests, not task-accuracy evidence."""
from copy import deepcopy
import argparse
from .io import *
from .geometry import *
from .model import *

def run(device):
 setup();rng=np.random.default_rng(42)
 s=np.array([[0,0,0],[1,0,0],[2,0,0],[1,1,0]],float);e=np.array([[0,1],[1,2],[1,3]])
 p=rng.normal(size=(180,3))*.7+[1,0,0];g=prepare(s,e,p);other=prepare(s*1.5,e,p*1.5)
 model=Model('h01','measure').to(device).eval()
 with torch.no_grad():
  single=model([g],[1]);pair=model([g,other],[1,1]);torch.testing.assert_close(single,pair[:1],atol=2e-6,rtol=2e-5)
  reverse=prepare(s,e[:,::-1],p[::-1]);torch.testing.assert_close(single,model([reverse],[1]),atol=2e-6,rtol=2e-5)
  perm=np.array([3,1,0,2]);inv=np.argsort(perm);gp=prepare(s[perm],inv[e],p)
  torch.testing.assert_close(single,model([gp],[int(inv[1])]),atol=2e-6,rtol=2e-5)
  # Degree-2 graph resampling cannot change the point grid, features, or masses.
  gr=prepare(np.r_[s,[[.5,0,0]]],np.array([[0,4],[4,1],[1,2],[1,3]]),p)
  for k in ('p','pf','c','w','mass'):torch.testing.assert_close(g[k],gr[k],atol=0,rtol=0)
  pure=prepare(np.empty((0,3)),np.empty((0,2),int),p)
  for k in ('p','pf','c','w','mass'):torch.testing.assert_close(g[k],pure[k],atol=0,rtol=0)
  # Duplicating input rows while splitting their mass preserves the whole tokenizer.
  dup=prepare(s,e,np.repeat(p,2,axis=0),weights=np.full(2*len(p),.5))
  torch.testing.assert_close(single,model([dup],[1]),atol=2e-6,rtol=2e-5)
  for arm in ('measure','count','wrong','skeleton','point','geometry'):
   m=Model('h01',arm).to(device);m.load_state_dict(model.state_dict());z=m([g],[1]);assert torch.isfinite(z).all()
  empty=prepare(s,e,np.empty((0,3)));assert torch.isfinite(model([empty],[1])).all()
  # Exact patch split under fixed score/value; compare both aggregation and null.
  op=model.encoder.integral;h=torch.randn(1,64,device=device);sf=torch.randn(3,64,device=device);mass=torch.tensor([1.,2.,5.],device=device)
  rel=dict(receiver=torch.zeros(3,device=device,dtype=torch.long),sender=torch.arange(3,device=device),donor=torch.arange(3,device=device),phi=torch.randn(3,12,device=device))
  u,n=op(h,sf,mass,rel);ix=torch.tensor([0,1,2,2],device=device);rel2=dict(receiver=torch.zeros(4,device=device,dtype=torch.long),sender=torch.arange(4,device=device),donor=torch.arange(4,device=device),phi=rel['phi'][ix]);mass2=torch.tensor([1.,2.,2.5,2.5],device=device)
  u2,n2=op(h,sf[ix],mass2,rel2);torch.testing.assert_close(u,u2,atol=2e-6,rtol=2e-5);torch.testing.assert_close(n,n2,atol=2e-6,rtol=2e-5)
  cu,cn=op(h,sf,mass,rel,'count');cu2,cn2=op(h,sf[ix],mass2,rel2,'count');assert (cu-cu2).norm()>1e-5
  stress=[]
  for fraction in (.25,.5,.75):
   ix=rng.choice(len(p),int(len(p)*fraction),replace=False);res=prepare(s,e,p[ix],weights=np.full(len(ix),len(p)/len(ix)))
   stress.append(dict(fraction=fraction,random_untrained_logit_l2=float((model([res],[1])-single).norm())))
 model.train();model([g],[1]).square().sum().backward()
 grads={name:float(sum((v.grad.square().sum() for v in mod.parameters() if v.grad is not None),torch.zeros((),device=device)).sqrt()) for name,mod in [('surface',model.encoder.surface),('integral',model.encoder.integral),('graph',model.encoder.graph_stem)]}
 assert all(v>0 and np.isfinite(v) for v in grads.values())
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 return dict(passed=True,device=device,checks=['packing_isolation','edge_reversal','node_permutation','point_only_tokenizer_equality','degree2_tokenizer_equality','empty_surface','all_arms_finite','empirical_point_split','fixed_patch_split','real_surface_gradients'],gradients=grads,count_split_delta=float((cu-cu2).norm()),resampling_stress=stress,parameters=sum(p.numel() for p in model.parameters()),limitation='Algebra invariant for identical split tokens only; resampling stress is untrained diagnostics, not accuracy.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();save(a.out,dict(cpu=run('cpu'),cuda=run('cuda')));print('CONTRACTS PASS')

"""Shared tensor schema for every task; observed geometry only enters encoding."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from .sampling import coarsen,representative_features,CONFIG

def knn_edges(a,b,k):
 _,idx=cKDTree(b).query(a,k=min(k,len(b)));idx=np.asarray(idx).reshape(len(a),-1)
 return np.repeat(np.arange(len(a)),idx.shape[1]),idx.ravel()

def construct(s,e,p,w,query):
 s=np.asarray(s,np.float64);p=np.asarray(p,np.float64);e=np.asarray(e,np.int64).reshape(-1,2);w=np.asarray(w,np.float64);query=np.asarray(query,np.float64).reshape(-1,3)
 assert len(s)>0 and len(p)>0 and np.isfinite(s).all() and np.isfinite(p).all()
 e=np.unique(np.sort(e,axis=1),axis=0);assert not len(e) or (e.min()>=0 and e.max()<len(s) and (e[:,0]!=e[:,1]).all())
 # Translation only; physical extent is never normalized away in retained metadata.
 origin=(p*w[:,None]).sum(0)/w.sum();s=s-origin;p=p-origin;query=query-origin
 cfg={k:v for k,v in CONFIG.items() if k in ['min_tokens','max_tokens','relative_rmse','max_relative_radius','grain_radius_factor','lloyd_steps']}
 c=coarsen(p,w,**cfg);pfeat,pintrinsic=representative_features(c);pos=c['pos'];pscale=c['scale'];pm=c['mass']
 degree=np.bincount(e.ravel(),minlength=len(s));length=np.linalg.norm(s[e[:,0]]-s[e[:,1]],axis=1);sm=np.zeros(len(s));total=np.zeros(len(s));direction=np.zeros_like(s)
 for j in [0,1]:
  np.add.at(sm,e[:,j],length*.5);np.add.at(total,e[:,j],length)
  delta=s[e[:,j]]-s[e[:,1-j]];delta/=np.maximum(length[:,None],1e-8);np.add.at(direction,e[:,j],delta)
 mean=total/np.maximum(degree,1);fallback=np.median(length[length>1e-8]) if (length>1e-8).any() else np.median(pscale)
 sh=np.maximum(mean,.1*max(fallback,1e-6));sm[sm==0]=max(fallback,1e-6)
 sf=np.column_stack((np.log1p(degree),np.log1p(mean),degree==1,degree>2,direction/np.maximum(degree[:,None],1),np.log1p(sh))).astype(np.float32)
 # Preserve every observed edge and add self support, never a thinned induced skeleton.
 ids=np.arange(len(s));ge=np.concatenate((e,e[:,::-1],np.column_stack((ids,ids))),0)
 sr,pj=knn_edges(s,pos,8);pr,sn=knn_edges(pos,s,4);cross=np.unique(np.column_stack((np.r_[sr,sn],np.r_[pj,pr])),axis=0)
 pr,pj=knn_edges(pos,pos,8);pe=np.column_stack((pr,pj))
 # A coarser common physical partition receives both modalities with separate total masses.
 mass=np.r_[sm/sm.sum(),pm/pm.sum()];joint=np.concatenate((s,pos));ncontext=min(CONFIG['context_tokens'],len(joint))
 coarse=coarsen(joint,mass,min_tokens=ncontext,max_tokens=ncontext,relative_rmse=0,max_relative_radius=0)
 cp=coarse['pos'];_,owner=cKDTree(cp).query(joint)
 scale=max(float(np.sqrt(((cp-cp.mean(0))**2).sum(1).mean())),1e-6)
 data=dict(spos=s.astype(np.float32),sf=sf,smass=sm.astype(np.float32),sscale=sh.astype(np.float32),graph_edge=ge.astype(np.int64),
  ppos=pos.astype(np.float32),pfeat=pfeat.astype(np.float16),pintrinsic=pintrinsic,pmass=pm.astype(np.float32),pscale=pscale.astype(np.float32),point_edge=pe.astype(np.int64),cross=cross.astype(np.int64),
  sowner=owner[:len(s)].astype(np.int64),powner=owner[len(s):].astype(np.int64),cpos=cp.astype(np.float32),cscale=np.array(scale,np.float32),query=query.astype(np.float32))
 assert abs(pm.sum()-w.sum())<=1e-8*w.sum()
 assert data['pfeat'].shape==(len(pos),16,4) and np.isfinite(data['pfeat']).all()
 return data,dict(c['stats'],skeleton_nodes=len(s),skeleton_edges=len(e),context_tokens=len(cp),cross_edges=len(cross),point_channels_zero_fraction=float((pfeat==0).all(-1).mean()),observation_scope='same frozen baseline point support; not complete original EM surface')

def pack(items,device='cuda'):
 """Disjoint-union batching. No attention/BN statistics cross sample identities."""
 out={k:[] for k in items[0] if k!='cscale'};sbase=pbase=cbase=0;sb=[];pb=[];cb=[];qbatch=[];sh=[];cs=[]
 for b,g in enumerate(items):
  ns,np_,nc=len(g['spos']),len(g['ppos']),len(g['cpos'])
  for k,v in g.items():
   if k=='cscale':continue
   if k=='graph_edge':v=v+sbase
   elif k=='point_edge':v=v+pbase
   elif k=='cross':v=v+torch.tensor([sbase,pbase])
   elif k in ['sowner','powner']:v=v+cbase
   out[k].append(v)
  sb.append(torch.full((ns,),b,dtype=torch.long));pb.append(torch.full((np_,),b,dtype=torch.long));cb.append(torch.full((nc,),b,dtype=torch.long));qbatch.append(torch.full((len(g['query']),),b,dtype=torch.long));cs.append(g['cscale']);sh.append((ns,np_,nc,len(g['query'])))
  sbase+=ns;pbase+=np_;cbase+=nc
 out={k:torch.cat(v).to(device) for k,v in out.items()};out.update(sbatch=torch.cat(sb).to(device),pbatch=torch.cat(pb).to(device),cbatch=torch.cat(cb).to(device),qbatch=torch.cat(qbatch).to(device),cscale=torch.stack(cs).to(device),shapes=sh,batch_size=len(items))
 return out

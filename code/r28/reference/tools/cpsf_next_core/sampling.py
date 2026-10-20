"""Adaptive measure coarsening, independent of labels and skeleton identities.

Weighted centroid split reduces within-cell squared distortion (variance identity).
The L sqrt(Q) bound applies to Lipschitz integrands, NOT downstream task risk.
"""
import heapq,time
import numpy as np
from scipy.spatial import cKDTree

CONFIG=dict(min_tokens=64,max_tokens=512,relative_rmse=.025,max_relative_radius=.12,neighbors=16,context_tokens=48,grain_radius_factor=4.,lloyd_steps=2)

def coarsen(points,mass,min_tokens=64,max_tokens=512,relative_rmse=.025,max_relative_radius=.12,grain_radius_factor=4.,lloyd_steps=2):
 p=np.asarray(points,np.float64);w=np.asarray(mass,np.float64);assert len(p)>0 and np.isfinite(p).all() and (w>0).all()
 # Input order/duplicate observations do not change the source measure.
 p,inv=np.unique(p,axis=0,return_inverse=True);w=np.bincount(inv,weights=w)
 total=w.sum();diag=max(np.linalg.norm(np.ptp(p,axis=0)),1e-8)
 probe=p[np.linspace(0,len(p)-1,min(1024,len(p))).astype(int)]
 dd,_=cKDTree(p).query(probe,k=min(16,len(p)))
 grain=max(float(np.median(np.asarray(dd).reshape(len(probe),-1)[:,-1])),1e-8)
 radius_limit=min(max_relative_radius*diag,grain_radius_factor*grain)
 nodes={};heap=[];counter=0
 def leaf(ids):
  nonlocal counter
  ww=w[ids];xx=p[ids];m=ww.sum();c=(xx*ww[:,None]).sum(0)/m;v=xx-c
  cov=(v*ww[:,None]).T@v/m;error=float(m*np.trace(cov));radius=float(np.linalg.norm(v,axis=1).max())
  node=dict(ids=ids,mass=m,center=c,cov=cov,error=error,radius=radius)
  key=counter;counter+=1;nodes[key]=node
  if len(ids)>1:
   # Largest measure-weighted error is refined first; radius constraint checked globally.
   heapq.heappush(heap,(-error,key))
  return key
 leaf(np.arange(len(p)));history=[];distortion=nodes[0]['error'];target_min=min(min_tokens,len(p));target_max=min(max_tokens,len(p))
 while len(nodes)<target_max:
  rmse=np.sqrt(max(distortion,0)/total)/diag;maxrad=max(x['radius'] for x in nodes.values())/diag
  if len(nodes)>=target_min and rmse<=relative_rmse and maxrad*diag<=radius_limit:break
  if not heap:break
  if len(nodes)>=target_min and maxrad*diag>radius_limit:
   key=max(nodes,key=lambda k:nodes[k]['radius'])
   if len(nodes[key]['ids'])<2:break
  else:
   while heap and heap[0][1] not in nodes:heapq.heappop(heap)
   if not heap:break
   _,key=heapq.heappop(heap)
  node=nodes.pop(key);ids=node['ids'];axis=np.linalg.eigh(node['cov'])[1][:,-1]
  # Fix eigenvector sign to make representation deterministic.
  if axis[np.argmax(abs(axis))]<0:axis=-axis
  order=np.argsort(p[ids]@axis,kind='stable');ids=ids[order];cs=np.cumsum(w[ids]);cut=int(np.searchsorted(cs,cs[-1]/2)+1);cut=max(1,min(cut,len(ids)-1))
  a=leaf(ids[:cut]);b=leaf(ids[cut:]);previous=distortion;distortion+=nodes[a]['error']+nodes[b]['error']-node['error']
  assert distortion<=previous+1e-6*max(1,previous)
  history.append(float(distortion/total))
 leaves=list(nodes.values());leaves.sort(key=lambda v:tuple(v['center']))
 owner=np.empty(len(p),np.int64)
 for j,node in enumerate(leaves):owner[node['ids']]=j
 centers=np.array([v['center'] for v in leaves]);m=np.array([v['mass'] for v in leaves]);cov=np.array([v['cov'] for v in leaves])
 before_lloyd=float(distortion/total)
 # Nearest-center reassignment plus weighted centroids never increases Q.
 for _ in range(lloyd_steps):
  _,assignment=cKDTree(centers).query(p);mass=np.bincount(assignment,weights=w,minlength=len(centers));new=np.zeros_like(centers);np.add.at(new,assignment,p*w[:,None]);valid=mass>0;new[valid]/=mass[valid,None];new[~valid]=centers[~valid]
  error=float(np.sum(w*((p-new[assignment])**2).sum(1)))
  assert error<=distortion+1e-6*max(1,distortion)
  centers=new;owner=assignment;distortion=error;m=mass
 # Remove empty numerical cells, preserving all observed mass.
 valid=m>0;remap=np.cumsum(valid)-1;owner=remap[owner];centers=centers[valid];m=m[valid]
 delta=p-centers[owner];cov=np.zeros((len(centers),3,3));np.add.at(cov,owner,w[:,None,None]*delta[:,:,None]*delta[:,None,:]);cov/=m[:,None,None]
 refined_radius=np.zeros(len(centers));np.maximum.at(refined_radius,owner,np.linalg.norm(delta,axis=1))
 scale=np.sqrt(np.trace(cov,axis1=1,axis2=2));radius=refined_radius
 # Neighbors come from real full observed support, never singleton-voxel offsets.
 dist,ix=cKDTree(p).query(centers,k=min(16,len(p)));dist=np.asarray(dist).reshape(len(centers),-1);ix=np.asarray(ix).reshape(len(centers),-1)
 support=(p[ix]*w[ix,None]).sum(1)/w[ix].sum(1)[:,None]
 local_scale=np.maximum(scale,np.sqrt(((p[ix]-support[:,None])**2).sum(-1).mean(1))).clip(1e-6)
 return dict(points=p,weights=w,owner=owner,pos=centers,mass=m,cov=cov,scale=local_scale,radius=radius,neighbor=ix,
  stats=dict(raw_points=len(inv),unique_points=len(p),tokens=len(centers),mass=float(total),relative_rmse=float(np.sqrt(max(distortion,0)/total)/diag),max_relative_radius=float(radius.max()/diag),distortion=float(max(distortion,0)/total),budget_limited=len(leaves)==target_max,monotonic_splits=len(history),reference_grain=grain,grain_radius_limit=radius_limit,pre_refinement_distortion=before_lloyd,lloyd_steps=lloyd_steps))

def representative_features(c):
 p,w=c['points'],c['weights'];ix=c['neighbor'];center=c['pos'];scale=c['scale'];near=p[ix];rel=(near-center[:,None])/scale[:,None,None]
 # Bounded local geometry remains in fixed physical axes. Physical scale is separate.
 rel=np.clip(rel,-12,12);d=np.linalg.norm(rel,axis=-1);x=np.concatenate((rel,d[...,None]),-1).astype(np.float32)
 if x.shape[1]<16:x=np.pad(x,((0,0),(0,16-x.shape[1]),(0,0)),mode='edge')
 covariance=c['cov']/scale[:,None,None]**2
 eig=np.maximum(np.linalg.eigvalsh(covariance),0)
 intrinsic=np.column_stack((eig,np.log1p(scale),np.log1p(c['radius']),np.log1p(c['mass']/c['mass'].mean()),covariance[:,0,1],covariance[:,0,2],covariance[:,1,2])).astype(np.float32)
 return x,intrinsic

def baseline_anchors(p,k,method):
 p=np.asarray(p,np.float64);k=min(k,len(p));rng=np.random.RandomState(42)
 if method=='random':return p[np.sort(rng.choice(len(p),k,replace=False))]
 if method=='uniform':
  # Adapt isotropic voxel width to the same occupied-grid budget; no first-row fill.
  extent=max(np.ptp(p,axis=0).max(),1e-8);lo=extent/1e6;hi=extent*2
  for _ in range(32):
   h=(lo+hi)/2;key=np.floor((p-p.min(0))/h).astype(np.int64);unique,inv=np.unique(key,axis=0,return_inverse=True)
   if len(unique)>k:lo=h
   else:hi=h
  h=lo;key=np.floor((p-p.min(0))/h).astype(np.int64);unique,inv=np.unique(key,axis=0,return_inverse=True);centers=(unique+.5)*h+p.min(0)
  best=np.full(len(unique),np.inf);ids=np.full(len(unique),-1,int)
  for i in range(len(p)):
   d=((p[i]-centers[inv[i]])**2).sum()
   if d<best[inv[i]]:best[inv[i]]=d;ids[inv[i]]=i
  if len(ids)>k:ids=ids[np.sort(rng.choice(len(ids),k,replace=False))]
  if len(ids)<k:ids=np.r_[ids,rng.choice(np.setdiff1d(np.arange(len(p)),ids),k-len(ids),replace=False)]
  return p[np.sort(ids)]
 if method=='fps':
  chosen=[];dist=np.full(len(p),np.inf);current=int(np.argmax(((p-p.mean(0))**2).sum(1)))
  for _ in range(k):
   chosen.append(current);dist=np.minimum(dist,((p-p[current])**2).sum(1));current=int(np.argmax(dist))
  return p[chosen]
 raise ValueError(method)

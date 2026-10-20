"""Frozen local-to-global transfer; train-only readout CV; explicit tile weights."""
import argparse,time
from .io import *
from .geometry import *
from .model import Model,mlp
from torch import nn

def even(n,k):return np.unique(np.linspace(0,n-1,min(n,k),dtype=int)) if n else np.empty(0,int)

def skeleton_thumbnail(s,e,limit):
 n=len(s)
 if n<=limit:keep=np.arange(n)
 else:
  selected=even(len(e),max(1,min(len(e),limit//2)));keep=np.unique(e[selected].ravel()) if len(selected) else np.empty(0,int)
  if len(keep)>limit:keep=keep[even(len(keep),limit)]
  if len(keep)<limit:
   degree=np.bincount(e.ravel(),minlength=n);anchors=np.setdiff1d(np.flatnonzero(degree!=2),keep);take=anchors[even(len(anchors),limit-len(keep))];remain=np.setdiff1d(np.arange(n),np.r_[keep,take]);fill=remain[even(len(remain),limit-len(keep)-len(take))];keep=np.sort(np.unique(np.r_[keep,take,fill]))
 lut=np.full(n,-1,int);lut[keep]=np.arange(len(keep));mask=(lut[e[:,0]]>=0)&(lut[e[:,1]]>=0)
 return s[keep],lut[e[mask]]

def sampled(s,e,p,limit,rng):
 ids=np.sort(rng.choice(len(p),min(len(p),limit),replace=False));weights=np.full(len(ids),len(p)/max(len(ids),1))
 return prepare(s,e,p[ids],weights=weights),dict(source_points=len(p),sample_points=len(ids),weight=float(weights[0]) if len(weights) else None)

def geometries(x,arm,mode='tiled'):
 rng=np.random.default_rng(42);p=x['surface_xyz_um'];meta=[]
 if arm=='point':
  empty=np.empty((0,3));edges=np.empty((0,2),int)
  glob,m=sampled(empty,edges,p,4096,rng);meta.append(m);tiles=[]
  if mode=='tiled':
   order=np.lexsort((p[:,2],p[:,1],p[:,0]));chunks=np.array_split(order,min(32,max(1,int(np.ceil(len(p)/16384)))))
   for ids in chunks:
    g,m=sampled(empty,edges,p[ids],2048,rng);tiles.append(g);meta.append(m)
  return [glob]+tiles,meta
 s=x['skeleton_xyz_um'];e=x['edge_index'].astype(int);owner=x['point_to_edge'].astype(int)
 ss,ee=skeleton_thumbnail(s,e,2048);glob,m=sampled(ss,ee,p,4096,rng);meta.append(m);tiles=[]
 if mode=='tiled' and len(e):
  mid=(s[e[:,0]]+s[e[:,1]])/2;order=np.lexsort((mid[:,2],mid[:,1],mid[:,0]));chunks=np.array_split(order,min(32,max(1,int(np.ceil(len(e)/4096)))))
  for core in chunks:
   core_nodes=np.unique(e[core].ravel());incident=np.flatnonzero(np.isin(e[:,0],core_nodes)|np.isin(e[:,1],core_nodes));nodes=np.unique(e[incident].ravel());ids=np.flatnonzero(np.isin(owner,core))
   if not len(ids):continue
   if len(nodes)>512:
    ordered=np.r_[core_nodes,np.setdiff1d(nodes,core_nodes)];nodes=ordered[even(len(ordered),512)]
   lut=np.full(len(s),-1,int);lut[nodes]=np.arange(len(nodes));mask=(lut[e[incident,0]]>=0)&(lut[e[incident,1]]>=0)
   g,m=sampled(s[nodes],lut[e[incident[mask]]],p[ids],2048,rng);tiles.append(g);meta.append(m)
 return [glob]+tiles,meta

@torch.no_grad()
def extract(checkpoint,out,smoke=False):
 setup();assert not out.exists();out.mkdir(parents=True);saved=load(checkpoint);arm=saved['protocol']['arm'];assert saved['protocol']['task']=='h01'
 model=Model('h01',arm).cuda();model.load_state_dict(saved['model']);model.eval()
 cells=sorted(json.loads((SOURCE/'manifest.json').read_text())['cells'],key=lambda r:r['cell'])
 if smoke:cells=[min((r for r in cells if r['split']==k),key=lambda r:r['surface_points']) for k in ('train','val')]
 records=[];features=[];ys=[];start=time.time()
 for row in cells:
  path=SOURCE/'inputs'/(row['cell']+'.npz')
  with np.load(path) as x:gs,metas=geometries(x,arm)
  vectors=[]
  for g in gs:
   if arm=='point':
    g['query']=relations(g['c'].numpy(),np.empty((0,2),int),g['c'].numpy(),Config(**g['config']));g['query_n']=g['m']
   elif arm=='skeleton':g=skeleton_view(g)
   packed,_=pack([g],'cuda');h,p,null=model.encoder(packed,arm)
   if arm=='point':hp=(h*packed['mass'][:,None]).sum(0)/packed['mass'].sum().clamp_min(1e-12)
   else:hp=h.mean(0) if len(h) else p.new_zeros(64)
   pp=(p*packed['mass'][:,None]).sum(0)/packed['mass'].sum().clamp_min(1e-12)
   vectors.append((hp.cpu(),pp.cpu(),float(packed['mass'].sum())))
  a,b,_=vectors[0]
  if len(vectors)>1:
   a=(a+torch.stack([v[0] for v in vectors[1:]]).mean(0))/2
   weights=torch.tensor([max(v[2],1) for v in vectors[1:]]);b=(b+(torch.stack([v[1] for v in vectors[1:]])*weights[:,None]).sum(0)/weights.sum())/2
  feature=torch.cat((a,b)).numpy();assert feature.shape==(128,) and np.isfinite(feature).all()
  with np.load(SOURCE/'targets'/(row['cell']+'.npz')) as y:target=float(y['excitatory_input_fraction'])
  features.append(feature);ys.append(target);records.append(dict(cell=row['cell'],split=row['split'],source_sha256=sha(path),tiles=len(gs)-1,sampling=metas))
  save(out/'status.json',dict(state='extracting',cells=len(records),expected=len(cells),seconds=time.time()-start));print('CELL',arm,len(records),row['cell'],flush=True)
 np.savez_compressed(out/'features.npz',x=np.asarray(features),y=np.asarray(ys),cell=np.asarray([r['cell'] for r in records]),split=np.asarray([r['split'] for r in records]))
 save(out/'manifest.json',dict(arm=arm,checkpoint_sha256=sha(checkpoint),encoder_frozen=True,scope='local-to-global frozen transfer',records=records,smoke=smoke,pooling='fixed thumbnail/local 1:1; mean node, empirical mass surface',point_partition='surface-only lexicographic chunks, ceil(P/16384) capped32; no graph access' if arm=='point' else None))
 save(out/'status.json',dict(state='extraction_complete',cells=len(records),seconds=time.time()-start))

def metrics(y,p):
 den=((y-y.mean())**2).sum();return dict(mae=float(abs(y-p).mean()),rmse=float(np.sqrt(((y-p)**2).mean())),r2=float(1-((y-p)**2).sum()/den) if den>0 else None,prediction=p.tolist())
def ridge(x,y,xe,alpha):
 mean=x.mean(0);std=np.maximum(x.std(0),.01);z=(x-mean)/std;ze=(xe-mean)/std
 u,s,vt=np.linalg.svd(z,full_matrices=False);coef=vt.T@((s/(s*s+alpha))*(u.T@(y-y.mean())))
 return np.clip(ze@coef+y.mean(),0,1),dict(coef=coef,mean=mean,std=std,intercept=np.asarray(y.mean()))
def fit_mlp(x,y,xe,wd,epochs):
 setup();x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32)[:,None];xe=torch.tensor(xe,dtype=torch.float32);mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.01);x=(x-mean)/std;xe=(xe-mean)/std
 head=nn.Sequential(mlp(128,64),nn.Linear(64,1),nn.Sigmoid());nn.init.zeros_(head[-2].weight);nn.init.constant_(head[-2].bias,float(torch.logit(y.mean().clamp(1e-6,1-1e-6))))
 opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=wd);pred={}
 for e in range(1,max(epochs)+1):
  opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(head(x),y);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);opt.step()
  if e in epochs:
   with torch.no_grad():pred[e]=head(xe).numpy()[:,0].copy()
 return pred,dict(head=head.state_dict(),mean=mean,std=std)

def fit(root):
 setup();data=np.load(root/'features.npz');train=np.flatnonzero(data['split']=='train');val=np.flatnonzero(data['split']=='val');assert len(train)==80 and len(val)==24
 ids=data['cell'];assert len(set(ids))==104;x=data['x'].astype(np.float64);y=data['y'];xt=x[train];yt=y[train]
 order=np.random.RandomState(42).permutation(80);folds=[(np.setdiff1d(np.arange(80),b),np.sort(b)) for b in np.array_split(order,5)]
 save(root/'folds.json',dict(seed=42,train_cells=ids[train].tolist(),val_cells=ids[val].tolist(),folds=[dict(train=a.tolist(),held=b.tolist()) for a,b in folds]));results={}
 scores=[]
 for alpha in [.01,.1,1,10,100,1000,10000]:
  oof=np.zeros(80)
  for a,b in folds:oof[b]=ridge(xt[a],yt[a],xt[b],alpha)[0]
  scores.append(dict(alpha=alpha,inner_oof_mae=float(abs(oof-yt).mean()),prediction=oof.tolist()))
 chosen=min(scores,key=lambda r:(r['inner_oof_mae'],-r['alpha']));pred,state=ridge(xt,yt,x[val],chosen['alpha']);np.savez_compressed(root/'ridge_weights.npz',**state)
 results['ridge']=dict(selected_alpha=chosen['alpha'],cv=scores,val=metrics(y[val],pred));save(root/'readout_progress.json',results)
 # Label isolation: outer validation does not enter this selection or fit.
 scores=[]
 for wd in [.01,.1,1.]:
  oof={e:np.zeros(80) for e in (25,50,100,200)}
  for a,b in folds:
   preds,_=fit_mlp(xt[a],yt[a],xt[b],wd,list(oof))
   for e,p in preds.items():oof[e][b]=p
  scores.extend([dict(weight_decay=wd,epochs=e,inner_oof_mae=float(abs(p-yt).mean()),prediction=p.tolist()) for e,p in oof.items()])
 chosen=min(scores,key=lambda r:(r['inner_oof_mae'],-r['weight_decay'],r['epochs']));pred,state=fit_mlp(xt,yt,x[val],chosen['weight_decay'],[chosen['epochs']]);torch.save(dict(**state,selected=chosen),root/'mlp_head.pt')
 results['mlp']=dict(selected=chosen,cv=scores,val=metrics(y[val],pred[chosen['epochs']]))
 save(root/'metrics.json',dict(seed=42,primary='ridge',scope='frozen encoder; train-only readout CV',manifest_sha256=sha(root/'manifest.json'),features_sha256=sha(root/'features.npz'),folds_sha256=sha(root/'folds.json'),results=results,constant=metrics(y[val],np.full(24,yt.mean())),val_cells=ids[val].tolist(),target=y[val].tolist()))
 save(root/'status.json',dict(state='complete'));print('READOUT COMPLETE',json.dumps({k:v['val']['mae'] for k,v in results.items()}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 try:
  extract(a.checkpoint,a.out,a.smoke)
  if not a.smoke:fit(a.out)
 except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

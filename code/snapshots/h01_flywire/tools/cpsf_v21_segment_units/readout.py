from .io import *
from .model import mlp
from torch import nn

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

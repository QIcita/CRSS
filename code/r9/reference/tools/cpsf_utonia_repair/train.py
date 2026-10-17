import argparse,time,hashlib,math
from collections import OrderedDict
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,average_precision_score
from torch.nn import functional as F
from .common import *
from .data import move
from .model import Model
class Cache:
 def __init__(self):self.items=OrderedDict();self.size=0
 def get(self,r):
  k=r['key']
  if k in self.items:
   v,n=self.items.pop(k);self.items[k]=(v,n);return v
  v=load_item(r);n=sum(z.numel()*z.element_size() for z in v.values() if isinstance(z,torch.Tensor))
  while self.items and self.size+n>512*2**20:
   _,(_,old)=self.items.popitem(last=False);self.size-=old
  if n<512*2**20:self.items[k]=(v,n);self.size+=n
  return v

def calc(y,z,task):
 y=np.asarray(y);z=np.asarray(z)
 if task=='h01_global':return dict(mae=float(abs(z-y).mean()),rmse=float(np.sqrt(((z-y)**2).mean())),r2=float(1-((z-y)**2).sum()/((y-y.mean())**2).sum()))
 pred=z.argmax(1)
 if task=='ted':
  tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum())
  return dict(f1=float(f1_score(y,pred,zero_division=0)),ap=float(average_precision_score(y,z[:,1])),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn,tn=tn)
 return dict(macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0)),accuracy=float(accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())
@torch.no_grad()
def evaluate(model,rr,cache,task,out):
 model.eval();z=[]
 for i,r in enumerate(rr):
  z.append(model(move(cache.get(r))).cpu().numpy())
  if i%100==0:save(out/'status.json',dict(state='evaluating_dev',processed=i+1,total=len(rr)))
 y=np.array([r['target'] for r in rr]);z=np.array(z);np.savez_compressed(out/'val_predictions.npz',keys=np.array([r['key'] for r in rr]),target=y,logits=z)
 return calc(y,z,task)
def run(task,kind):
 setup();p=verify();cfg=p['tasks'][task];out=ROOT/'results'/f'{kind}_{task}';out.mkdir(parents=True,exist_ok=True)
 gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
 if (out/'metrics.json').exists():return
 rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];assert (len(tr),len(va))==(cfg['train'],cfg['val']);assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
 for r in rr:
  assert sha(r['path'])==r['sha256'];assert sha(r['feature_path'])==r['feature_sha256']
 m=Model(task,kind).cuda();initial=state_sha(m);mean=float(np.mean([r['target'] for r in tr]));scale=max(float(np.std([r['target'] for r in tr])),.05) if task=='h01_global' else 1.
 if task=='h01_global':
  with torch.no_grad():m.head.bias.fill_(float(np.log(mean/(1-mean))))
 identity=dict(version=ROOT.name,task=task,kind=kind,seed=42,protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'data'/task/'manifest.json'),initial_state_sha256=initial,pretrained=False,test_used=False,selection='fixed final epoch',official_modules=p['upstream'])
 save(out/'identity.json',identity)
 if task=='ted':opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:1/(e+1))
 else:opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr_min'])
 start=time.time();steps=0;cache=Cache();history=[];active=set();start_epoch=1
 if (out/'latest.pt').exists():
  c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert c['identity']==identity;m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);history=c['history'];steps=c['steps'];start_epoch=c['epoch']+1;np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
 else:torch.save(m.state_dict(),out/'initial.pt')
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,cfg['epochs']+1):
  m.train();begin=time.time();lr=opt.param_groups[0]['lr']
  if task=='ted':
   blocks=sorted({r['source']['block'] for r in tr});lookup={b:[] for b in blocks}
   for r in tr:lookup[r['source']['block']].append(r)
   rng=np.random.RandomState(42+epoch);order=[r for i in rng.permutation(len(blocks)) for r in lookup[blocks[i]]]
  else:order=[tr[i] for i in np.random.permutation(len(tr))]
  total=0.
  for lo in range(0,len(order),cfg['batch']):
   group=order[lo:lo+cfg['batch']];opt.zero_grad(set_to_none=True)
   for r in group:
    z=m(move(cache.get(r)));y=torch.tensor(r['target'],device='cuda')
    if task=='ted':loss=F.binary_cross_entropy_with_logits(z,F.one_hot(y.long(),2).float())*(5 if r['target'] else 1);objective=loss
    elif task=='h01_global':loss=((z-y.float())/scale).square();objective=loss/len(group)
    else:loss=F.cross_entropy(z[None],y.long()[None],label_smoothing=.05);objective=loss/len(group)
    assert torch.isfinite(loss);objective.backward();total+=float(loss.detach())
   assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
   if lo==0:active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
   if task!='ted':torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
   opt.step();steps+=1
   if steps%5==1:save(out/'status.json',dict(state='training',epoch=epoch,epochs=cfg['epochs'],processed=lo+len(group),total=len(order),steps=steps,elapsed_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
  sched.step();rec=dict(epoch=epoch,lr=lr,loss=total/len(order),seconds=time.time()-begin,steps=steps,sample_order_sha256=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest());history.append(rec);save(out/'history.json',history);print(json.dumps(rec),flush=True)
  tmp=out/'latest.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,steps=steps,history=history,identity=identity,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(out/'latest.pt')
 assert steps==cfg['updates'];torch.save(dict(model=m.state_dict(),identity=identity),out/'final.pt');metrics=evaluate(m,va,cache,task,out)
 with np.load(out/'val_predictions.npz') as a:assert metrics==calc(a['target'],a['logits'],task)
 save(out/'metrics.json',dict(identity=identity,val=metrics,epochs=cfg['epochs'],steps=steps,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),final_sha256=sha(out/'final.pt'),independent_metric_check=True,active_gradient_parameters=sorted(active),test_used=False));save(out/'status.json',dict(state='complete',test_used=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--kind',required=True);a=a.parse_args()
 try:run(a.task,a.kind)
 except BaseException as e:save(ROOT/'results'/f'{a.kind}_{a.task}'/'failure.json',dict(error=repr(e)));raise

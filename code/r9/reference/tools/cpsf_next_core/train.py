import argparse,time,math
from collections import OrderedDict
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,average_precision_score
from torch.nn import functional as F
from .common import *
from .data import pack
from .model import Model

class Cache:
 def __init__(self,budget=512*2**20):self.items=OrderedDict();self.size=0;self.budget=budget
 def get(self,row):
  key=row['key']
  if key in self.items:
   v,n=self.items.pop(key);self.items[key]=(v,n);return v
  v=load_item(row);n=sum(x.numel()*x.element_size() for x in v.values())
  while self.items and self.size+n>self.budget:
   _,(_,old)=self.items.popitem(last=False);self.size-=old
  if n<=self.budget:self.items[key]=(v,n);self.size+=n
  return v

def calc(y,z,task):
 y=np.asarray(y);z=np.asarray(z)
 if task=='h01_global':return dict(mae=float(abs(z-y).mean()),rmse=float(np.sqrt(((z-y)**2).mean())),r2=float(1-((z-y)**2).sum()/((y-y.mean())**2).sum()))
 if task=='ted':
  pred=z.argmax(1);tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum())
  return dict(f1=float(f1_score(y,pred,zero_division=0)),ap=float(average_precision_score(y,z[:,1])),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn,tn=tn)
 pred=z.argmax(1);return dict(macro_f1=float(f1_score(y,pred,average='macro',labels=list(range(3 if task=='h01_local' else 6)),zero_division=0)),accuracy=float(accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=list(range(3 if task=='h01_local' else 6))).tolist())

@torch.no_grad()
def evaluate(model,records,cache,task,out,micro):
 model.eval();zs=[]
 for start in range(0,len(records),micro):
  batch=records[start:start+micro];g=pack([cache.get(r) for r in batch]);zs.append(model(g).cpu().numpy())
  if start%(micro*20)==0:save(out/'status.json',dict(state='evaluating_dev',processed=start+len(batch),total=len(records)))
 z=np.concatenate(zs);y=np.array([r['target'] for r in records]);np.savez_compressed(out/'val_predictions.npz',keys=np.array([r['key'] for r in records]),target=y,logits=z)
 return calc(y,z,task)

def run(task,arm):
 setup();protocol=verify();cfg=protocol['tasks'][task];rr=rows(task);out=ROOT/'results'/f'{task}_{arm}';out.mkdir(parents=True,exist_ok=True)
 gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
 if (out/'metrics.json').exists():return
 tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];assert (len(tr),len(va))==(cfg['train'],cfg['val']);assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
 for r in rr:assert sha(r['path'])==r['sha256']
 model=Model(task,arm).cuda();initsha=state_sha(model);mean=float(np.mean([r['target'] for r in tr]));scale=max(float(np.std([r['target'] for r in tr])),.05) if task=='h01_global' else 1.
 if task=='h01_global':
  with torch.no_grad():model.head.bias.fill_(float(np.log(mean/(1-mean))));model.head.weight.mul_(.01)
 identity=dict(version=ROOT.name,task=task,arm=arm,seed=42,protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'data'/task/'manifest.json'),initial_state_sha256=initsha,pretrained=False,test_used=False,selection='fixed final epoch')
 save(out/'identity.json',identity)
 if task=='ted':
  optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:1/(e+1))
 else:
  optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'],eta_min=cfg['lr_min'])
 history=[];steps=0;start_epoch=1;start=time.time();cache=Cache();active=set()
 if (out/'latest.pt').exists():
  ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert ck['identity']==identity;model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);schedule.load_state_dict(ck['scheduler']);history=ck['history'];steps=ck['steps'];start_epoch=ck['epoch']+1;np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 else:torch.save(model.state_dict(),out/'initial.pt')
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,cfg['epochs']+1):
  model.train();t=time.time()
  if task=='ted':
   blocks=sorted({r['source']['block'] for r in tr});lookup={b:[] for b in blocks}
   for r in tr:lookup[r['source']['block']].append(r)
   rng=np.random.RandomState(42+epoch);order=[r for i in rng.permutation(len(blocks)) for r in lookup[blocks[i]]]
  else:order=[tr[i] for i in np.random.permutation(len(tr))]
  total=0.;lr=optimizer.param_groups[0]['lr']
  for lo in range(0,len(order),cfg['batch']):
   batch=order[lo:lo+cfg['batch']];optimizer.zero_grad(set_to_none=True)
   for j in range(0,len(batch),cfg['micro']):
    part=batch[j:j+cfg['micro']];g=pack([cache.get(r) for r in part]);z=model(g);target=torch.tensor([r['target'] for r in part],device='cuda')
    if task=='ted':
     yy=F.one_hot(target.long(),2).float();losses=F.binary_cross_entropy_with_logits(z,yy,reduction='none').mean(1)*torch.where(target>0,5.,1.);loss=losses.sum()
    elif task=='h01_global':losses=((z-target.float())/scale).square();loss=losses.sum()/len(batch)
    else:losses=F.cross_entropy(z,target.long(),label_smoothing=.05,reduction='none');loss=losses.sum()/len(batch)
    assert torch.isfinite(loss);loss.backward();total+=float(losses.detach().sum())
   assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
   if lo==0:active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
   if task!='ted':torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
   optimizer.step();steps+=1
   if steps%10==1:save(out/'status.json',dict(state='training',epoch=epoch,epochs=cfg['epochs'],processed=lo+len(batch),total=len(order),steps=steps,elapsed_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
  schedule.step();rec=dict(epoch=epoch,lr=lr,loss=total/len(order),seconds=time.time()-t,steps=steps,sample_order_sha256=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest());history.append(rec);save(out/'history.json',history);print(json.dumps(rec),flush=True)
  tmp=out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),identity=identity,epoch=epoch,steps=steps,history=history,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(out/'latest.pt')
 assert steps==cfg['updates'];torch.save(dict(model=model.state_dict(),identity=identity),out/'final.pt');metrics=evaluate(model,va,cache,task,out,cfg['micro'])
 # Reload prediction artifact for independent metric aggregation.
 with np.load(out/'val_predictions.npz') as a:independent=calc(a['target'],a['logits'],task)
 assert metrics==independent
 save(out/'metrics.json',dict(identity=identity,val=metrics,independent_metric_check=True,epochs=cfg['epochs'],steps=steps,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),final_sha256=sha(out/'final.pt'),active_gradient_parameters=sorted(active),test_used=False))
 save(out/'status.json',dict(state='complete',test_used=False))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--arm',default='full');a=parser.parse_args()
 try:run(a.task,a.arm)
 except BaseException as e:save(ROOT/'results'/f'{a.task}_{a.arm}'/'failure.json',dict(error=repr(e)));raise

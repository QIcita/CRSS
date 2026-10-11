"""Executable task recipe. Full data READY gate required; never uses provisional splits."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import OrderedDict,Counter
import numpy as np
import torch
from .task import CellModel,read_views,require_ready
from .evaluation import classify,retrieve
from .build import save,sha
from tools.cpsf_v19_measure_core.io import setup

class Cache:
 def __init__(self,root,arm):self.root=root;self.arm=arm;self.items=OrderedDict();self.size=0
 @staticmethod
 def tensor_bytes(value):
  if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
  if isinstance(value,dict):return sum(Cache.tensor_bytes(v) for v in value.values())
  if isinstance(value,(list,tuple)):return sum(Cache.tensor_bytes(v) for v in value)
  return 0
 def get(self,row):
  key=row['root_id']
  if key in self.items:
   value,size=self.items.pop(key);self.items[key]=(value,size);return value
  file=self.root/'samples'/(key+'.npz');assert sha(file)==row['sample_sha256'];value=read_views(file,self.arm);size=self.tensor_bytes(value)
  while self.items and self.size+size>1024*2**20:
   _,(_,old)=self.items.popitem(last=False);self.size-=old
  if size<=1024*2**20:self.items[key]=(value,size);self.size+=size
  return value

@torch.no_grad()
def evaluate(model,rows,cache,target,nclasses,split,out):
 model.eval();ys=[];features=[];logits=[]
 for i,r in enumerate(rows):
  z=model.embedding(cache.get(r));features.append(z.cpu().numpy());logits.append(model.head(z).cpu().numpy());ys.append(r[target])
  if (i+1)%100==0:print('EVAL',split,i+1,len(rows),flush=True)
 x=np.asarray(features);z=np.asarray(logits);y=np.asarray(ys);np.savez_compressed(out/(split+'_features.npz'),x=x,logits=z,target=y,root_id=np.asarray([r['root_id'] for r in rows]),cell_type=np.asarray([r['cell_type'] for r in rows]))
 return classify(y,z,nclasses),x,z

def run(a):
 setup();assert a.seed==42;rows,mapping=require_ready(a.root);cfg=json.loads((a.root/'metadata/configs'/(a.task+'.json')).read_text());target=cfg['target'];classes=len(mapping[a.task]);sets={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
 ready=json.loads((a.root/'dataset/READY.json').read_text());assert sha(a.root/'dataset/frozen_manifest.jsonl')==ready['manifest_sha256'] and sha(a.root/'dataset/labels.json')==ready['labels_sha256']
 assert torch.cuda.is_available(),'Formal training requires a profiled GPU resource slot'
 a.out.mkdir(parents=True);model=CellModel(classes,a.arm).cuda();cache=Cache(a.root,a.arm);epochs=cfg['training']['epochs'];opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003)
 protocol=dict(config=cfg,arm=a.arm,seed=42,classes=classes,registered_parameters=sum(p.numel() for p in model.parameters()),ready=ready,config_sha256=sha(a.root/'metadata/configs'/(a.task+'.json')),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},encoder_source_sha256={p.name:sha(p) for p in (Path(__file__).parent.parent/'cpsf_v19_measure_core').glob('*.py')},resume=False)
 save(a.out/'protocol.json',protocol);history=[];started=time.time()
 for epoch in range(1,epochs+1):
  model.train();order=np.random.permutation(len(sets['train']));losses=[];begin=time.time()
  for lo in range(0,len(order),4):
   group=[sets['train'][i] for i in order[lo:lo+4]];opt.zero_grad(set_to_none=True);loss_value=0.
   for r in group:
    logits=model(cache.get(r));loss=torch.nn.functional.cross_entropy(logits,torch.tensor([r[target]],device='cuda'))/len(group);assert torch.isfinite(loss);loss.backward();loss_value+=float(loss.detach())
   torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(loss_value)
  schedule.step();record=dict(epoch=epoch,epochs=epochs,loss=float(np.mean(losses)),seconds=time.time()-begin);history.append(record);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**record));print(json.dumps(record),flush=True)
  tmp=a.out/'latest.tmp';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,protocol=protocol),tmp);tmp.replace(a.out/'latest.pt')
 torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol=protocol),a.out/'final.pt');save(a.out/'status.json',dict(state='evaluating'))
 metrics={};features={};logits={}
 for split in ('val','test'):
  metrics[split],features[split],logits[split]=evaluate(model,sets[split],cache,target,classes,split,a.out)
 if a.task=='fine':
  _,features['train'],_=evaluate(model,sets['train'],cache,target,classes,'train',a.out);freq=Counter(r['cell_type'] for r in rows)
  for split in ('val','test'):
   rs=sets[split];y=np.asarray([r['fine_target'] for r in rs]);p=logits[split].argmax(1);pc=metrics[split]['per_class'];strata={}
   for name,lo,hi in [('tail',5,19),('middle',20,99),('head',100,10**9)]:
    ids=[i for i,t in enumerate(mapping['fine']) if lo<=freq[t]<=hi];mask=np.isin(y,ids);strata[name]=dict(cells=int(mask.sum()),classes=len(ids),accuracy=float((p[mask]==y[mask]).mean()) if mask.any() else None,macro_f1=float(np.mean([pc[i]['f1'] for i in ids])) if ids else None)
   metrics[split]['long_tail']=strata
   parent=np.asarray(mapping['fine_to_coarse']);coarse_target=np.asarray([r['coarse_target'] for r in rs]);metrics[split]['coarse_accuracy_from_fine_argmax']=float((parent[p]==coarse_target).mean())
   retrieval=retrieve(features[split],features['train'],[r['cell_type'] for r in rs],[r['cell_type'] for r in sets['train']],[r['root_id'] for r in rs],[r['root_id'] for r in sets['train']]);save(a.out/(split+'_retrieval.json'),retrieval)
 save(a.out/'metrics.json',dict(protocol=protocol,results=metrics,seconds=time.time()-started));save(a.out/'status.json',dict(state='complete'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--task',choices=['coarse','fine'],required=True);p.add_argument('--arm',choices=['measure','count','point','skeleton','geometry'],required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable run output already exists')
 try:run(a)
 except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

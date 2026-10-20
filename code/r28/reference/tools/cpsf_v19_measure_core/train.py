"""Frozen task protocols; final checkpoint only, no validation selection."""
import argparse,time
from .io import *
from .data import h01_sets,ted_sets,geometry
from .model import Model
from torch import nn

@torch.no_grad()
def evaluate(model,rows,cache,task,out,split):
 model.eval();records=[];ys=[];ps=[]
 blocks=[dict(block='h01',events=rows)] if task=='h01' else rows
 for b in blocks:
  logits=[];targets=[]
  for start in range(0,len(b['events']),8):
   rs=b['events'][start:start+8];gs=[geometry(r,cache,task) for r in rs]
   qs=[load(r['path'])['query'] for r in rs] if task=='h01' else None
   z=model(gs,qs).cpu();logits.append(z);targets.extend([r['target'] if task=='h01' else int(r['target'].argmax()) for r in rs])
  z=torch.cat(logits);pred=z.argmax(1) if task=='h01' else z.sigmoid().round().argmax(1)
  ys.extend(targets);ps.extend(pred.tolist())
  records.extend([dict(sample=r.get('sample',r.get('name')),cell=r.get('cell'),target=int(y),prediction=int(p),logits=v) for r,y,p,v in zip(b['events'],targets,pred.tolist(),z.tolist())])
  if task=='ted':
   np.savez_compressed(out/f'{split}_{Path(b["block"]).stem}.npz',logits=z.numpy(),target=np.asarray(targets))
   print('EVAL',split,b['block'],len(ys),flush=True)
 return dict(**(classification(ys,ps) if task=='h01' else counts(ys,ps)),samples=len(ys),rows=records)

def run(a):
 setup();assert a.seed==42;assert not a.out.exists();a.out.mkdir(parents=True);start=time.time()
 if a.task=='h01':sets=h01_sets(a.cache,a.smoke);audit=[]
 else:sets,audit=ted_sets(a.smoke);save(a.out/'unit_audit.json',dict(unit='native_swc; um calibration not verified',blocks=audit))
 model=Model(a.task,a.arm).cuda();cache=Cache(disk=Path(__file__).resolve().parents[2]/'cache_ted_geometry' if a.task=='ted' else None);epochs=2 if a.smoke else (35 if a.task=='h01' else 30)
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001) if a.task=='h01' else torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003) if a.task=='h01' else torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1/(epoch+1))
 cohort={k:[{key:r[key] for key in (('sample','cell','target','sha256') if a.task=='h01' else ('block','source_sha256'))} for r in seq] for k,seq in sets.items()}
 protocol=dict(version='V19 measure core',task=a.task,arm=a.arm,seed=42,epochs=epochs,smoke=a.smoke,selection='fixed final epoch',registered_parameters=sum(p.numel() for p in model.parameters()),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).resolve().parents[2]/'CPSF_V19_MEASURE_CORE_PROTOCOL_20260930.md'),cohort=cohort,cohort_sha256=hashlib.sha256(json.dumps(cohort,sort_keys=True).encode()).hexdigest(),train_count=len(sets['train']) if a.task=='h01' else sum(len(b['events']) for b in sets['train']),eval_count=len(sets['val']) if a.task=='h01' else sum(len(b['events']) for b in sets['test']),resume=False)
 save(a.out/'protocol.json',protocol);history=[];opt.zero_grad(set_to_none=True)
 for epoch in range(1,epochs+1):
  model.train();begin=time.time();losses=[];grad=[];wrong=[]
  def batch(rs,update):
   gs=[geometry(r,cache,a.task) for r in rs];qs=[load(r['path'])['query'] for r in rs] if a.task=='h01' else None
   # Bound autograd memory while preserving summed gradients and optimizer batches.
   micro=32 if a.task=='ted' else 4
   for lo in range(0,len(rs),micro):
    sub=rs[lo:lo+micro];z=model(gs[lo:lo+micro],None if qs is None else qs[lo:lo+micro])
    if a.task=='h01':
     y=torch.tensor([r['target'] for r in sub],device='cuda');loss=nn.functional.cross_entropy(z,y,reduction='sum')/len(rs)
    else:
     y=torch.stack([r['target'] for r in sub]).cuda();loss=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none').mean(1)*torch.where(y[:,1]>0,5.,1.)).sum()
    assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach()))
   if a.arm=='wrong':wrong.extend([float((g['donor']!=g['sender']).float().mean()) for g in gs if len(g['sender'])])
   if update:
    if a.task=='h01':nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    grad.append({name:float(sum((p.grad.double().square().sum() for p in module.parameters() if p.grad is not None),torch.zeros((),device='cuda')).sqrt()) for name,module in [('surface',model.encoder.surface),('integral',model.encoder.integral),('graph',model.encoder.graph_stem)]})
    opt.step();opt.zero_grad(set_to_none=True)
  if a.task=='h01':
   order=np.random.permutation(len(sets['train']))
   for lo in range(0,len(order),8):batch([sets['train'][i] for i in order[lo:lo+8]],True)
  else:
   pending=[]
   for idx in torch.randperm(len(sets['train'])).tolist():
    pending.extend(sets['train'][idx]['events'])
    while len(pending)>=64:batch(pending[:64],True);pending=pending[64:]
   if pending:batch(pending,False)
  sched.step();row=dict(epoch=epoch,epochs=epochs,seconds=time.time()-begin,loss=float(np.mean(losses)),gradient={k:float(np.mean([r[k] for r in grad])) for k in ('surface','integral','graph')},wrong_fraction=float(np.mean(wrong)) if wrong else None)
  history.append(row);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**row));print(json.dumps(row),flush=True)
  tmp=a.out/'latest.tmp';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,protocol=protocol),tmp);tmp.replace(a.out/'latest.pt')
 torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol=protocol),a.out/'final.pt')
 save(a.out/'status.json',dict(state='evaluating',epoch=epochs));key='val' if a.task=='h01' else 'test';metrics={key:evaluate(model,sets[key],cache,a.task,a.out,key)}
 if a.task=='h01':metrics['train']=evaluate(model,sets['train'],cache,a.task,a.out,'train')
 save(a.out/'metrics.json',dict(protocol=protocol,**metrics,seconds=time.time()-start));save(a.out/'status.json',dict(state='complete',seconds=time.time()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['h01','ted'],required=True);p.add_argument('--arm',choices=['measure','count','skeleton','point','geometry','wrong'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 try:run(a)
 except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

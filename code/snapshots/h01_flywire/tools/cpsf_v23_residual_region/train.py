import argparse,time,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup,save,load,sha,classification,Cache
from .model import Model,ARMS
ROOT=Path(__file__).resolve().parents[2]

def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in g.items()}

def evaluate(model,rows,cache):
    predictions=[];ys=[]
    model.eval()
    with torch.no_grad():
        for r in rows:
            g=move(cache.get(r['path'],lambda r=r:load(r['path'])));y=model(g).cpu()
            predictions.append(int(y.argmax()) if model.task=='local' else float(y));ys.append(r['target'])
    if model.task=='local':metrics=classification(ys,predictions)
    else:
        y=np.array(ys);p=np.array(predictions);den=((y-y.mean())**2).sum()
        metrics=dict(mae=float(abs(p-y).mean()),rmse=float(np.sqrt(((p-y)**2).mean())),r2=float(1-((p-y)**2).sum()/den) if den else None)
    return dict(**metrics,rows=[dict(cell=r['cell'],key=r['key'],target=y,prediction=p) for r,y,p in zip(rows,ys,predictions)])

def run(a):
    setup();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert protocol['seed']==42
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,(name,'source drift')
    assert not a.out.exists();a.out.mkdir(parents=True)
    cache_root=ROOT/('smoke_cache' if a.smoke else 'cache')
    ready=json.loads((cache_root/'READY.json').read_text());assert sha(cache_root/'manifest.json')==ready['manifest_sha256']
    m=json.loads((cache_root/'manifest.json').read_text());modal='joint' if a.arm in ('full','mean_only','no_fusion') else a.arm
    rows=[r for r in m['records'] if r['task']==a.task and r['modal']==modal]
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='val']
    assert not ({r['cell'] for r in training}&{r['cell'] for r in validation})
    if a.smoke:validation=training
    else:
        assert len({r['cell'] for r in training})==80 and len({r['cell'] for r in validation})==24
        assert (len(training),len(validation))==((716,214) if a.task=='local' else (80,24))
    for r in training+validation:assert sha(r['path'])==r['sha256']
    model=Model(a.task,a.arm).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initial,a.out/'initial.pt')
    if a.task=='global':
        mean=float(np.mean([r['target'] for r in training]));scale=max(float(np.std([r['target'] for r in training])),.05)
        with torch.no_grad():model.head.bias.fill_(float(np.log(np.clip(mean,1e-5,1-1e-5)/(1-np.clip(mean,1e-5,1-1e-5)))))
    else:mean=0.;scale=1.
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    epochs=2 if a.smoke else protocol['epochs'][a.task]
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00003)
    cache=Cache(budget=768*2**20);history=[];start=time.time();batch=8 if a.task=='local' else 4
    torch.cuda.reset_peak_memory_stats();active=set()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.permutation(len(training));loss_sum=0.;t=time.time()
        for begin in range(0,len(order),batch):
            group=[training[i] for i in order[begin:begin+batch]];optimizer.zero_grad(set_to_none=True)
            for row in group:
                g=move(cache.get(row['path'],lambda row=row:load(row['path'])));pred=model(g)
                if a.task=='local':loss=F.cross_entropy(pred[None],torch.tensor([row['target']],device='cuda'),label_smoothing=.05)
                else:loss=((pred-row['target'])/scale).square()
                assert torch.isfinite(loss);(loss/len(group)).backward();loss_sum+=float(loss.detach())
                del g,pred,loss
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        active.update(n for n,p in model.named_parameters() if p.grad is not None and bool((p.grad!=0).any()))
        schedule.step();rec=dict(epoch=epoch,epochs=epochs,mean_loss=loss_sum/len(training),seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(rec);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**rec));print(json.dumps(rec),flush=True)
        tmp=a.out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,task=a.task,arm=a.arm,seed=42,protocol_sha256=sha(ROOT/'protocol.json')),tmp);tmp.replace(a.out/'latest.pt')
    assert any(n.startswith('scale.') for n in active) and any(n.startswith('position.') for n in active),'Scale/position not trained'
    val=evaluate(model,validation,cache);train=evaluate(model,training,cache)
    metrics=dict(task=a.task,arm=a.arm,seed=42,epochs=epochs,smoke=a.smoke,encoder_frozen=False,
                 evaluation='task-supervised; final fixed epoch; development split; not frozen transfer',val=val,train=train,
                 train_target_mean=mean,train_target_scale=scale,train_cells=sorted({r['cell'] for r in training}),val_cells=sorted({r['cell'] for r in validation}),
                 protocol_sha256=sha(ROOT/'protocol.json'),cache_manifest_sha256=ready['manifest_sha256'],seconds=time.time()-start,
                 registered_parameters=sum(p.numel() for p in model.parameters()),active_gradient_parameters=sorted(active),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    torch.save(dict(model=model.state_dict(),epoch=epochs,metrics=metrics),a.out/'final.pt');save(a.out/'metrics.json',metrics);save(a.out/'status.json',dict(state='complete',epochs=epochs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['local','global'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

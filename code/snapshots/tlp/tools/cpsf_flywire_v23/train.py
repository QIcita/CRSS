import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_flywire_20260930.evaluation import classify,retrieve
from .data import ROOT,load
from .model import Model,ARMS,variant

def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in g.items()}

def dataset(arm,profile=False):
    root=ROOT/('profile_cache' if profile else 'cache')
    ready=json.loads((root/'READY.json').read_text());assert ready['ready']
    assert sha(root/'manifest.json')==ready['manifest_sha256']
    m=json.loads((root/'manifest.json').read_text());rows=[r for r in m['records'] if r['variant']==variant(arm)]
    return rows,m['labels'],ready

@torch.no_grad()
def evaluate(model,rows,cache,task,classes,out,split):
    model.eval();features=[];logits=[];target=[]
    for i,row in enumerate(rows):
        g=move(cache.get(row['path'],lambda row=row:load(row['path'])))
        z=model.embedding(g);features.append(z.cpu().numpy());logits.append(model.head(z).cpu().numpy());target.append(row[task+'_target'])
    x=np.asarray(features);z=np.asarray(logits);y=np.asarray(target)
    np.savez_compressed(out/(split+'_features.npz'),x=x,logits=z,target=y,root_id=np.array([r['root_id'] for r in rows]))
    return classify(y,z,classes),x,z

def add_fine(metrics,x,z,rows,gallery,gallery_x,labels,allrows,out,split):
    freq=Counter(r['cell_type'] for r in allrows);y=np.array([r['fine_target'] for r in rows]);pred=z.argmax(1)
    strata={}
    for name,lo,hi in [('tail',5,19),('middle',20,99),('head',100,10**9)]:
        ids=[i for i,t in enumerate(labels['fine']) if lo<=freq[t]<=hi];mask=np.isin(y,ids)
        strata[name]=dict(classes=len(ids),cells=int(mask.sum()),macro_f1=float(np.mean([metrics['per_class'][i]['f1'] for i in ids])) if ids else None,
            accuracy=float((pred[mask]==y[mask]).mean()) if mask.any() else None)
    metrics['long_tail']=strata
    r=retrieve(x,gallery_x,[r['cell_type'] for r in rows],[r['cell_type'] for r in gallery],
        [r['root_id'] for r in rows],[r['root_id'] for r in gallery]);save(out/(split+'_retrieval.json'),r)
    metrics['retrieval']={k:v for k,v in r.items() if k!='rows'}

def run(a):
    setup();protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,(name,'source drift')
    assert not a.out.exists();a.out.mkdir(parents=True)
    rows,labels,ready=dataset(a.arm);classes=len(labels[a.task]);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    assert len(train)==4975 and len(val)==1573
    assert not ({r['root_id'] for r in train}&{r['root_id'] for r in val})
    for r in train+val:assert sha(r['path'])==r['sha256']
    model=Model(classes,a.arm).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(initial,a.out/'initial.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,protocol['epochs'],eta_min=protocol['lr_min'])
    cache=Cache(budget=512*2**20);history=[];start=time.time();active=set();torch.cuda.reset_peak_memory_stats()
    save(a.out/'status.json',dict(state='training',epoch=0,epochs=protocol['epochs']))
    for epoch in range(1,protocol['epochs']+1):
        model.train();order=np.random.permutation(len(train));loss_sum=0.;t=time.time()
        for begin in range(0,len(order),protocol['batch']):
            group=[train[i] for i in order[begin:begin+protocol['batch']]];opt.zero_grad(set_to_none=True)
            for r in group:
                g=move(cache.get(r['path'],lambda r=r:load(r['path'])));z=model(g)
                loss=F.cross_entropy(z[None],torch.tensor([r[a.task+'_target']],device='cuda'),label_smoothing=protocol['label_smoothing'])
                assert torch.isfinite(loss);(loss/len(group)).backward();loss_sum+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip'],error_if_nonfinite=True);opt.step()
            if begin%400==0:save(a.out/'status.json',dict(state='training',epoch=epoch,epochs=protocol['epochs'],processed=min(begin+len(group),len(train)),train_cells=len(train)))
        active.update(n for n,p in model.named_parameters() if p.grad is not None and bool((p.grad!=0).any()))
        schedule.step();rec=dict(epoch=epoch,epochs=protocol['epochs'],mean_loss=loss_sum/len(train),seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(rec);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**rec));print(json.dumps(rec),flush=True)
        tmp=a.out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha256=sha(ROOT/'protocol.json')),tmp);tmp.replace(a.out/'latest.pt')
    assert any(n.startswith('scale.') for n in active)==model.use_scale
    assert any(n.startswith('position.') for n in active)==model.use_position
    torch.save(dict(model=model.state_dict(),task=a.task,arm=a.arm,epoch=protocol['epochs'],protocol_sha256=sha(ROOT/'protocol.json')),a.out/'final.pt')
    save(a.out/'status.json',dict(state='evaluating_validation',test_used=False))
    metrics,x,z=evaluate(model,val,cache,a.task,classes,a.out,'val')
    if a.task=='fine':
        _,gallery_x,_=evaluate(model,train,cache,a.task,classes,a.out,'train')
        add_fine(metrics,x,z,val,train,gallery_x,labels,rows,a.out,'val')
    save(a.out/'metrics.json',dict(task=a.task,arm=a.arm,seed=42,epochs=protocol['epochs'],val=metrics,test_used=False,
        seconds=time.time()-start,active_gradient_parameters=sorted(active),cache_manifest_sha256=ready['manifest_sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),final_sha256=sha(a.out/'final.pt'),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    save(a.out/'status.json',dict(state='complete',test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['coarse','fine'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

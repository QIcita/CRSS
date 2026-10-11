"""Ordinary CE only; model, cache, initialization, order and update budget match expanded balanced controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_flywire_20260930.evaluation import classify
from tools.cpsf_flywire_v23.data import load
from .common import ROOT,COHORT,ARMS,identity
from tools.cpsf_flywire_balanced.loss import balanced_loss,weights_from_counts
from tools.cpsf_flywire_coarse_cohort.budget import select_round,order_hash
from tools.cpsf_flywire_v23.model import Model,variant

def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in g.items()}

def dataset(arm,profile=False):
    root=COHORT/'cache'
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

def load_reference(path,digest):
    assert sha(path)==digest
    return torch.load(path,map_location='cpu',weights_only=False)

def run(a):
    setup();protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,(name,'source drift')
    assert not a.out.exists();a.out.mkdir(parents=True)
    rows,labels,ready=dataset(a.arm);assert ready['manifest_sha256']==protocol['cache_manifest_sha256'];classes=len(labels[a.task]);train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    assert len(train)==protocol['counts']['train'] and len(val)==protocol['counts']['val']
    assert sum(r['legacy'] for r in val)==1573
    assert not ({r['root_id'] for r in train}&{r['root_id'] for r in val})
    counts=[sum(r['coarse_target']==i for r in train) for i in range(classes)]
    assert counts==protocol['loss']['train_counts']
    class_weights=torch.ones(classes,device='cuda')
    assert class_weights.cpu().tolist()==protocol['loss']['class_weights']
    if a.smoke:
        assert sha(ROOT/'smoke_records.json')==protocol['smoke_records_sha256']
        train=json.loads((ROOT/'smoke_records.json').read_text());val=train
        assert all(r['split']=='train' for r in train)
    for r in train+val:assert sha(r['path'])==r['sha256']
    model=Model(classes,a.arm).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(initial,a.out/'initial.pt')
    reference=torch.load(Path(protocol['reference_runs'][a.arm]['path'])/'initial.pt',map_location='cpu',weights_only=False)
    assert initial.keys()==reference.keys() and all(torch.equal(v,reference[k]) for k,v in initial.items())
    run_identity=dict(identity(protocol,a.arm),run_id=ROOT.name+'/'+str(a.out.relative_to(ROOT)),
        smoke=a.smoke,optimizer_steps=2*((len(train)+protocol['batch']-1)//protocol['batch']) if a.smoke else protocol['optimizer_steps'])
    save(a.out/'identity.json',dict(run_identity,smoke=a.smoke))
    epochs=2 if a.smoke else protocol['epochs']
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=protocol['lr_min'])
    cache=Cache(budget=512*2**20);history=[];start=time.time();active=set();seen=set();total_steps=0;torch.cuda.reset_peak_memory_stats()
    save(a.out/'status.json',dict(state='training',epoch=0,epochs=epochs))
    for epoch in range(1,epochs+1):
        model.train();order=select_round(train,np.random,len(train) if a.smoke else protocol['epoch_cells']);seen.update(train[i]['root_id'] for i in order);loss_sum=0.;raw_loss_sum=0.;clipped=0;steps=0;t=time.time()
        for begin in range(0,len(order),protocol['batch']):
            group=[train[i] for i in order[begin:begin+protocol['batch']]];opt.zero_grad(set_to_none=True)
            for r in group:
                g=move(cache.get(r['path'],lambda r=r:load(r['path'])));z=model(g)
                target=torch.tensor([r[a.task+'_target']],device='cuda')
                raw=F.cross_entropy(z[None],target,label_smoothing=protocol['label_smoothing'])
                loss=balanced_loss(z[None],target,class_weights,protocol['label_smoothing'])
                raw_loss_sum+=float(raw.detach())
                assert torch.isfinite(loss);(loss/len(group)).backward();loss_sum+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip'],error_if_nonfinite=True);clipped+=int(norm>protocol['clip']);steps+=1;total_steps+=1;opt.step()
            if begin%400==0:save(a.out/'status.json',dict(state='training',epoch=epoch,epochs=epochs,processed=min(begin+len(group),len(train)),train_cells=len(train)))
        active.update(n for n,p in model.named_parameters() if p.grad is not None and bool((p.grad!=0).any()))
        schedule.step();rec=dict(epoch=epoch,epochs=epochs,sample_order_sha256=order_hash([train[i] for i in order]),round_cells=len(order),mean_loss=loss_sum/len(order),unweighted_mean_loss=raw_loss_sum/len(order),gradient_clip_fraction=clipped/steps,seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(rec);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**rec));print(json.dumps(rec),flush=True)
        tmp=a.out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha256=sha(ROOT/'protocol.json'),identity=run_identity),tmp);tmp.replace(a.out/'latest.pt')
    if not a.smoke:
        assert total_steps==protocol['optimizer_steps'] and len(seen)==len(train)
    assert any(n.startswith('scale.') for n in active)==model.use_scale
    assert any(n.startswith('position.') for n in active)==model.use_position
    torch.save(dict(model=model.state_dict(),task=a.task,arm=a.arm,epoch=epochs,protocol_sha256=sha(ROOT/'protocol.json'),identity=run_identity),a.out/'final.pt')
    save(a.out/'status.json',dict(state='evaluating_validation',test_used=False))
    metrics,x,z=evaluate(model,val,cache,a.task,classes,a.out,'val')
    subset_metrics={}
    if not a.smoke:
        y=np.array([r['coarse_target'] for r in val]);legacy=np.array([r['legacy'] for r in val])
        subset_metrics=dict(val_legacy=classify(y[legacy],z[legacy],classes),val_added=classify(y[~legacy],z[~legacy],classes))
        del model;torch.cuda.empty_cache()
        reference=Model(classes,a.arm).cuda();ref=protocol['reference_runs'][a.arm];checkpoint=load_reference(Path(ref['path'])/'final.pt',ref['sha256']['final.pt'])
        reference.load_state_dict(checkpoint['model'])
        refdir=a.out/'frozen_reference';refdir.mkdir();refmetrics,rx,rz=evaluate(reference,val,cache,a.task,classes,refdir,'val')
        assert abs(refmetrics['macro_f1']-ref['val']['macro_f1'])<1e-8,'Expanded frozen reference reproduction mismatch'
        oldval=classify(y[legacy],rz[legacy],classes)
        assert abs(oldval['macro_f1']-ref['val_legacy']['macro_f1'])<1e-8,'Frozen reference reproduction mismatch'
        save(refdir/'metrics.json',dict(val=refmetrics,val_legacy=oldval,val_added=classify(y[~legacy],rz[~legacy],classes),test_used=False,source_final_sha256=ref['sha256']['final.pt'],source_run=ref['path'],evaluation_role='frozen weighted-CE reference; not current CE run'))
    save(a.out/'metrics.json',dict(task=a.task,arm=a.arm,seed=42,epochs=epochs,smoke=a.smoke,identity=run_identity,class_weights=class_weights.cpu().tolist(),validation_kind='training-subset engineering smoke' if a.smoke else 'expanded development validation',val=metrics,test_used=False,**subset_metrics,total_optimizer_steps=total_steps,unique_train_cells_seen=len(seen),
        seconds=time.time()-start,active_gradient_parameters=sorted(active),cache_manifest_sha256=ready['manifest_sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),final_sha256=sha(a.out/'final.pt'),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    save(a.out/'status.json',dict(state='complete',test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['coarse'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

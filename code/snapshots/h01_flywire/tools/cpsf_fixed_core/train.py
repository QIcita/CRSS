import argparse,json,time,math
import numpy as np
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup,Cache
from tools.cpsf_flywire_20260930.evaluation import classify
from .common import *
from .model import Model

def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in g.items()}
def smoke_rows(train,task):
    if task=='h01_global':return [max(train,key=lambda r:Path(r['path']).stat().st_size)]
    chosen=[next(r for r in train if r['target']==c) for c in range(6 if task=='flywire_coarse' else 3)]
    chosen.append(max(train,key=lambda r:Path(r['path']).stat().st_size))
    return list({r['key']:r for r in chosen}.values())
def expected_initial(task):
    p=verify();ref=p['references'][task+'/full']
    assert sha(Path(ref['path'])/'initial.pt')==ref['hashes']['initial.pt']
    return torch.load(Path(ref['path'])/'initial.pt',map_location='cpu',weights_only=False)

@torch.no_grad()
def evaluate(model,rr,cache,task,out,label):
    model.eval();z=[];gates=[]
    for r in rr:
        g=move(cache.get(r['path'],lambda r=r:load_graph(r,task)));z.append(model(g).cpu().numpy())
    z=np.asarray(z);y=np.asarray([r['target'] for r in rr]);ids=np.array([r['key'] for r in rr])
    np.savez_compressed(out/(label+'_predictions.npz'),key=ids,target=y,prediction_or_logits=z)
    if task=='h01_global':
        den=((y-y.mean())**2).sum();return dict(mae=float(abs(z-y).mean()),rmse=float(np.sqrt(((z-y)**2).mean())),r2=float(1-((z-y)**2).sum()/den) if den else None)
    return classify(y,z,6 if task=='flywire_coarse' else 3)

def run(a):
    setup();p=verify();rr,ready=rows(a.task,a.arm);training=[r for r in rr if r['split']=='train'];val=[r for r in rr if r['split']=='val']
    settings=p['tasks'][a.task];assert len(training)==settings['train'] and len(val)==settings['val']
    assert not ({r['cell'] for r in training}&{r['cell'] for r in val})
    if a.smoke:training=smoke_rows(training,a.task);val=training
    assert not a.out.exists();a.out.mkdir(parents=True)
    for r in training+val:assert sha(r['path'])==r['sha256']
    model=Model(a.task,a.arm).cuda();reference=expected_initial(a.task)
    state=model.state_dict();inherited=state
    assert all(k in reference and torch.equal(v.cpu(),reference[k]) for k,v in inherited.items())
    torch.save({k:v.detach().cpu() for k,v in state.items()},a.out/'initial.pt')
    identity=dict(experiment_id=ROOT.name,run_id=ROOT.name+'/'+str(a.out.relative_to(ROOT)),task=a.task,arm=a.arm,
        model_role='retained fixed residual; gate-free; coefficient 0.1',
        hierarchy=p['hierarchy'][a.task],seed=42,smoke=a.smoke,protocol_sha256=sha(ROOT/'protocol.json'),
        data_manifest_sha256=ready['manifest_sha256'],source_sha256=digest(p['source_sha256']),test_used=False)
    save(a.out/'identity.json',identity)
    if a.task=='h01_global':
        mean=float(np.mean([r['target'] for r in training]));scale=max(float(np.std([r['target'] for r in training])),.05)
        with torch.no_grad():model.head.bias.fill_(float(np.log(np.clip(mean,1e-5,1-1e-5)/(1-np.clip(mean,1e-5,1-1e-5)))))
    else:mean=0.;scale=1.
    op=p['optimizer'];opt=torch.optim.AdamW(model.parameters(),lr=op['lr'],weight_decay=op['weight_decay'])
    epochs=2 if a.smoke else settings['epochs'];batch=settings['batch'];schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=op['lr_min'])
    cache=Cache(budget=512*2**20);history=[];active=set();steps=0;seen=set();start=time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.permutation(len(training))
        if not a.smoke:order=order[:settings['round_cells']]
        seen.update(training[i]['key'] for i in order);t=time.time();loss_sum=0.;clipped=0
        for lo in range(0,len(order),batch):
            group=[training[i] for i in order[lo:lo+batch]];opt.zero_grad(set_to_none=True)
            for r in group:
                g=move(cache.get(r['path'],lambda r=r:load_graph(r,a.task)));z=model(g)
                loss=((z-r['target'])/scale).square() if a.task=='h01_global' else F.cross_entropy(z[None],torch.tensor([r['target']],device='cuda'),label_smoothing=.05)
                assert torch.isfinite(loss);(loss/len(group)).backward();loss_sum+=float(loss.detach())
            active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),op['clip'],error_if_nonfinite=True);clipped+=int(norm>op['clip']);opt.step();steps+=1
            if lo%400==0:save(a.out/'status.json',dict(state='training',epoch=epoch,epochs=epochs,processed=lo+len(group),round_cells=len(order)))
        schedule.step();rec=dict(epoch=epoch,mean_loss=loss_sum/len(order),seconds=time.time()-t,
            sample_order_sha256=__import__('hashlib').sha256('\n'.join(training[i]['key'] for i in order).encode()).hexdigest(),
            residual_coefficient=0.1,gradient_clip_fraction=clipped/math.ceil(len(order)/batch))
        history.append(rec);save(a.out/'history.json',history);print(json.dumps(rec),flush=True)
        tmp=a.out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),epoch=epoch,
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),identity=identity),tmp);tmp.replace(a.out/'latest.pt')
    if not a.smoke:assert steps==settings['updates'] and len(seen)==len(training)
    assert any(n.startswith('position.') for n in active)==model.use_position
    assert any(n.startswith('scale.') for n in active)==model.use_scale
    assert any(n.startswith('graph.') for n in active)==(a.arm!='point')
    assert any(n.startswith('point.') for n in active)==(a.arm!='skeleton')
    save(a.out/'status.json',dict(state='evaluating_validation',test_used=False))
    valmetrics=evaluate(model,val,cache,a.task,a.out,'val');subsets={}
    if a.task=='flywire_coarse' and not a.smoke:
        subsets={label:evaluate(model,[r for r in val if r['legacy']==legacy],cache,a.task,a.out,label) for label,legacy in [('val_legacy',True),('val_added',False)]}
    torch.save(dict(model=model.state_dict(),identity=identity,epoch=epochs),a.out/'final.pt')
    m=dict(identity=identity,task=a.task,arm=a.arm,seed=42,smoke=a.smoke,epochs=epochs,test_used=False,val=valmetrics,**subsets,
        train_target_mean=mean,train_target_scale=scale,total_optimizer_steps=steps,unique_train_samples_seen=len(seen),unique_train_cells_seen=len({r['cell'] for r in training}),
        protocol_sha256=sha(ROOT/'protocol.json'),final_sha256=sha(a.out/'final.pt'),active_gradient_parameters=sorted(active),
        residual_coefficient=0.1,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    if not a.smoke and a.arm=='full':
        key='mae' if a.task=='h01_global' else 'macro_f1';old=p['references'][a.task+'/full']['val'][key]
        assert abs(valmetrics[key]-old)<1e-7,('baseline reproduction mismatch',a.task,valmetrics[key],old)
    save(a.out/'metrics.json',m);save(a.out/'status.json',dict(state='complete',test_used=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

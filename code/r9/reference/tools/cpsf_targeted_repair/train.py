import argparse,time,hashlib,json
import numpy as np
from scipy.stats import rankdata
from torch.nn import functional as F
from .common import *
from .model import Model
from tools.cpsf_missing_edge.objective import missing_edge_loss
from tools.cpsf_v21_segment_units.io import counts
def state_digest(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def average_precision(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float)
    if not y.any():return 0.
    order=np.argsort(-s,kind='stable');ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    tp=np.cumsum(y[order])[ends]
    return float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*tp/(ends+1)))
def auc(y,s):
    y=np.asarray(y,bool);pos=int(y.sum());neg=len(y)-pos
    return float((rankdata(s)[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None

@torch.no_grad()
def evaluate(model,records,cache,out,task):
    model.eval();all_y=[];all_s=[];blocks=[]
    for i,r in enumerate(records):
        item=cache.get(r['path'],lambda r=r:load(r['path']));g=move(item['g']);z=model(g).cpu().numpy();y=item['target'].numpy()
        if task=='ted':all_y.append(int(y.argmax()));all_s.append(z)
        else:
            score=torch.from_numpy(z).sigmoid().numpy();obs=item['observed'].numpy();upper=np.triu(np.ones_like(obs),1).astype(bool);mask=upper&~obs
            assert np.max(abs(score-score.T))<1e-6
            yy=y[mask]>0;ss=score[mask];all_y.append(yy);all_s.append(ss)
            off=~np.eye(len(y),dtype=bool);full_y=y[off]>0;full_s=score[off]
            pairs=item['pairs'].numpy();cy=y[pairs[:,0],pairs[:,1]]>0;cs=score[pairs[:,0],pairs[:,1]]
            blocks.append(dict(key=r['key'],missing_edge_ap=average_precision(yy,ss),missing_positives=int(yy.sum()),unobserved_pairs=len(yy),
                full_matrix_ap=average_precision(full_y,full_s),full_matrix_auc=auc(full_y,full_s),
                candidate_ap=average_precision(cy,cs),max_asymmetry=float(abs(score-score.T).max())))
            np.savez_compressed(out/('prediction_'+r['key']+'.npz'),score=score,target=y,observed=obs,pairs=pairs)
        if i%100==0:save(out/'status.json',dict(state='evaluating_dev',processed=i+1,total=len(records)))
    if task=='ted':
        z=np.array(all_s);pred=torch.tensor(z).sigmoid().round().argmax(1).numpy()
        np.savez_compressed(out/'dev_predictions.npz',keys=np.array([r['key'] for r in records]),target=np.array(all_y),logits=z)
        return dict(**counts(all_y,pred),events=len(records),ap=average_precision(all_y,z[:,1]))
    # Independent recomputation from saved predictions, excluding all observed edges.
    audit=[]
    for r in records:
        with np.load(out/('prediction_'+r['key']+'.npz')) as a:
            ix=np.triu_indices(len(a['target']),1);keep=~a['observed'][ix].astype(bool)
            audit.append(average_precision(a['target'][ix][keep]>0,a['score'][ix][keep]))
    macro=float(np.mean([x['missing_edge_ap'] for x in blocks]));assert abs(macro-float(np.mean(audit)))<1e-12
    save(out/'independent_missing_edge_audit.json',dict(passed=True,macro_ap=macro,blocks=len(records),universe='all unordered unobserved pairs, including noncandidate pairs'))
    return dict(missing_edge_macro_ap=macro,missing_edge_micro_ap=average_precision(np.concatenate(all_y),np.concatenate(all_s)),
        full_matrix_macro_ap=float(np.mean([x['full_matrix_ap'] for x in blocks])),blocks=blocks)

def group_metrics(task,rr,out):
    groups=sorted({r['group'] for r in rr});result={}
    if task=='ted':
        with np.load(out/'dev_predictions.npz') as z:
            yy=z['target'];score=z['logits'];pred=torch.tensor(score).sigmoid().round().argmax(1).numpy()
        for group in groups:
            ix=np.array([r['group']==group for r in rr]);result[group]=dict(**counts(yy[ix],pred[ix]),ap=average_precision(yy[ix],score[ix,1]),events=int(ix.sum()))
    else:
        for group in groups:
            aps=[]
            for r in rr:
                if r['group']!=group:continue
                with np.load(out/('prediction_'+r['key']+'.npz')) as z:
                    mask=np.triu(~z['observed'].astype(bool),1);aps.append(average_precision(z['target'][mask]>0,z['score'][mask]))
            result[group]=dict(missing_edge_macro_ap=float(np.mean(aps)),graphs=len(aps))
    save(out/'dev_group_metrics.json',result)
    return result

def run(task,arm):
    setup();p=verify();out=ROOT/'results'/f'{task}_{arm}';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    if (out/'metrics.json').exists():return
    rr=records(task);training=[r for r in rr if r['split']=='train'];dev=[r for r in rr if r['split']=='dev']
    assert training and dev and {r['group'] for r in training}.isdisjoint(r['group'] for r in dev)
    for r in rr:assert sha(r['path'])==r['sha256']
    side_manifest=ROOT/'manifests'/f'{task}_sidecars.json';side=json.loads(side_manifest.read_text())
    assert side['complete'] and side['protocol_sha256']==sha(ROOT/'protocol.json')
    if arm=='repair':
        for r in side['records']:assert sha(r['path'])==r['sha256']
    cfg=p['tasks'][task];epochs=cfg['epochs'];batch=cfg['batch'];model=Model(task,arm).cuda();initial=state_digest(model)
    identity=dict(task=task,arm=arm,seed=42,protocol_sha256=sha(ROOT/'protocol.json'),data_manifest_sha256=sha(ROOT/'manifests'/f'{task}.json'),side_manifest_sha256=sha(side_manifest),initial_state_sha256=initial,experiment_id=ROOT.name,test_used=False,historical_val_used=False,checkpoint_selection='fixed final epoch')
    save(out/'identity.json',identity)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1/(epoch+1)) if task=='ted' else torch.optim.lr_scheduler.StepLR(opt,20,.1)
    history=[];steps=0;begin=time.time();cache=RepairCache(rr,task,arm);start_epoch=1
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert ck['identity']==identity
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        history=ck['history'];steps=ck['steps'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:torch.save(model.state_dict(),out/'initial.pt')
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,epochs+1):
        model.train();rng=np.random.RandomState(42+epoch)
        if task=='ted':
            blocks=sorted({r['block'] for r in training});lookup={b:[] for b in blocks}
            for r in training:lookup[r['block']].append(r)
            order=[r for i in rng.permutation(len(blocks)) for r in lookup[blocks[i]]]
        else:order=[training[i] for i in rng.permutation(len(training))]
        t=time.time();total=0.;active=set();epoch_lr=opt.param_groups[0]['lr']
        for lo in range(0,len(order),batch):
            group=order[lo:lo+batch];opt.zero_grad(set_to_none=True)
            for r in group:
                item=cache.get(r['path']);g=move(item['g']);y=item['target'].cuda();z=model(g)
                loss=F.binary_cross_entropy_with_logits(z,y)*(5 if y[1]>0 else 1) if task=='ted' else missing_edge_loss(z,y,item['observed'].cuda())
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
            assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters())
            if lo<4*batch:active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            opt.step();steps+=1
            if lo%(batch*10)==0:save(out/'status.json',dict(state='training',epoch=epoch,epochs=epochs,processed=lo+len(group),samples=len(order),optimizer_steps=steps,elapsed_seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        sched.step();rec=dict(epoch=epoch,lr=epoch_lr,loss=total/len(order),seconds=time.time()-t,optimizer_steps=steps,sample_order_sha256=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest(),active_gradient_parameters=sorted(active))
        history.append(rec);save(out/'history.json',history);print(json.dumps({k:v for k,v in rec.items() if k!='active_gradient_parameters'}),flush=True)
        tmp=out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,history=history,steps=steps,identity=identity,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(out/'latest.pt')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity,epoch=epochs),out/'final.pt')
    metrics=evaluate(model,dev,cache,out,task);groups=group_metrics(task,dev,out)
    save(out/'metrics.json',dict(identity=identity,dev=metrics,dev_groups=groups,epochs=epochs,optimizer_steps=steps,seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated(),final_sha256=sha(out/'final.pt'),test_used=False,historical_val_used=False))
    save(out/'status.json',dict(state='complete',test_used=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['ted','tlp']);p.add_argument('--arm',required=True,choices=['control','repair']);a=p.parse_args()
    try:run(a.task,a.arm)
    except BaseException as e:save(ROOT/'results'/f'{a.task}_{a.arm}'/'failure.json',dict(error=repr(e)));raise

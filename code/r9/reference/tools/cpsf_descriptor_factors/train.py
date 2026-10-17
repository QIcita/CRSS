import argparse,time,math
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
        if i%100==0:save(out/'status.json',dict(state='evaluating_val',processed=i+1,total=len(records)))
    if task=='ted':
        z=np.array(all_s);pred=torch.tensor(z).sigmoid().round().argmax(1).numpy()
        np.savez_compressed(out/'val_predictions.npz',keys=np.array([r['key'] for r in records]),target=np.array(all_y),logits=z)
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

def run(task,arm,descriptor,smoke=False):
    setup();p=verify();out=ROOT/('smoke' if smoke else 'results')/job_key(arm,descriptor);out.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'contracts/tlp.json').read_text())['protocol_sha256']==sha(ROOT/'protocol.json')
    if (out/'metrics.json').exists():return
    if smoke:
        path=DATA_PARENT/'pilot'/task/modal(arm)/'manifest.json';rr=json.loads(path.read_text())['records'];datahash=sha(path)
    else:rr,datahash=rows(task,arm)
    training=[r for r in rr if r['split']=='train'];val=training if smoke else [r for r in rr if r['split']=='val']
    assert training and val
    if not smoke:assert {r['group'] for r in training}.isdisjoint(r['group'] for r in val)
    for r in training+val:assert sha(r['path'])==r['sha256']
    cfg=p['tasks'][task];epochs=2 if smoke else cfg['epochs'];batch=cfg['batch'];model=Model(task,arm,descriptor).cuda()
    initial=state_digest(model)
    parent_identity=json.loads((parent_result(arm,descriptor)/'identity.json').read_text())
    assert initial==parent_identity['initial_state_sha256']
    identity=dict(task=task,arm=arm,seed=42,smoke=smoke,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=digest(p['source_sha256']),data_manifest_sha256=datahash,
        initial_state_sha256=initial,experiment_id=ROOT.name,descriptor=descriptor,model_family='retained deterministic fixed core; descriptor channel factors stage5',objective_id=p['objective_id'],parent_protocol_sha256=p['parent_protocol_sha256'],test_used=False)
    save(out/'identity.json',identity)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1/(epoch+1)) if task=='ted' else torch.optim.lr_scheduler.StepLR(opt,20,.1)
    history=[];steps=0;begin=time.time();cache=Cache(budget=1024*2**20);start_epoch=1
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
            blocks=sorted({r['block'] for r in training});order=[r for i in rng.permutation(len(blocks)) for r in training if r['block']==blocks[i]]
        else:order=[training[i] for i in rng.permutation(len(training))]
        t=time.time();total=0.;active=set()
        for lo in range(0,len(order),batch):
            group=order[lo:lo+batch];opt.zero_grad(set_to_none=True)
            for r in group:
                item=cache.get(r['path'],lambda r=r:load(r['path']));g=move(item['g']);y=item['target'].cuda();z=model(g)
                if task=='ted':loss=F.binary_cross_entropy_with_logits(z,y)*(5 if y[1]>0 else 1)
                else:
                    loss=missing_edge_loss(z,y,item['observed'].cuda())
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
            assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters())
            if lo<4*batch:active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            opt.step();steps+=1
            if lo%(batch*10)==0:save(out/'status.json',dict(state='training',epoch=epoch,epochs=epochs,processed=lo+len(group),samples=len(order),optimizer_steps=steps))
        sched.step();rec=dict(epoch=epoch,loss=total/len(order),seconds=time.time()-t,optimizer_steps=steps,
            sample_order_sha256=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest(),active_gradient_parameters=sorted(active))
        history.append(rec);save(out/'history.json',history);print(json.dumps({k:v for k,v in rec.items() if k!='active_gradient_parameters'}),flush=True)
        tmp=out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,history=history,steps=steps,identity=identity,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(out/'latest.pt')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity,epoch=epochs),out/'final.pt')
    valmetrics=evaluate(model,val,cache,out,task)
    save(out/'metrics.json',dict(identity=identity,val=valmetrics,epochs=epochs,optimizer_steps=steps,seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        final_sha256=sha(out/'final.pt'),test_used=False,smoke=smoke))
    save(out/'status.json',dict(state='complete',test_used=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--arm',required=True,choices=ARMS);p.add_argument('--descriptor',required=True,choices=DESCRIPTORS);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.task,a.arm,a.descriptor,a.smoke)
    except BaseException as e:save(ROOT/('smoke' if a.smoke else 'results')/job_key(a.arm,a.descriptor)/'failure.json',dict(error=repr(e)));raise

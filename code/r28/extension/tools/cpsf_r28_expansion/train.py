import argparse,time,math,json,hashlib
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,average_precision_score
from tools.cpsf_raw_point_attention.common import ROOT,setup,verify,sha,save,rows,state_sha
from tools.cpsf_missing_edge.objective import missing_edge_loss
from .model import Model
from .data import load_item,move

class Cache:
    def __init__(self):self.items=OrderedDict();self.size=0
    def get(self,r):
        k=r['key']
        if k in self.items:v,n=self.items.pop(k);self.items[k]=(v,n);return v
        v=load_item(r);n=sum(z.numel()*z.element_size() for z in v['g'].values() if isinstance(z,torch.Tensor))
        while self.items and self.size+n>512*2**20:
            _,(_,old)=self.items.popitem(last=False);self.size-=old
        if n<512*2**20:self.items[k]=(v,n);self.size+=n
        return v

def optimizer(m,task,cfg):
    if task in ('ted','tlp'):
        opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:1/(e+1)) if task=='ted' else torch.optim.lr_scheduler.StepLR(opt,20,.1)
    else:
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr_min'])
    return opt,sched

def loss_for(z,item,row,task):
    if task=='tlp':return missing_edge_loss(z,item['target'].cuda(),item['observed'].cuda())
    y=torch.tensor(row['target'],device='cuda')
    if task=='ted':return F.binary_cross_entropy_with_logits(z,F.one_hot(y.long(),2).float())*(5 if row['target'] else 1)
    return F.cross_entropy(z[None],y.long()[None],label_smoothing=.05)

def calc(y,z,task):
    pred=z.argmax(1)
    if task=='ted':
        tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum())
        return dict(f1=float(f1_score(y,pred,zero_division=0)),ap=float(average_precision_score(y,z[:,1])),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn,tn=tn)
    return dict(macro_f1=float(f1_score(y,pred,average='macro',labels=list(range(6)),zero_division=0)),accuracy=float(accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=list(range(6))).tolist())

def eval_forward(model,g,task):
    if task!='tlp':return model(g)
    # Same accepted repair as R9: demand exact symmetric pair features;
    # preserve original upper FP32 logits, mirror only the unused lower half.
    reads=[];offset=0
    def read_hook(module,args,out):reads.append(out)
    def check_pair(module,args):
        nonlocal offset
        feat=args[0];context,q=reads[-2:];a=q[None];b=q[offset:offset+len(feat),None]
        reverse=torch.cat((a+b,(a-b).abs(),context[g['pair_lookup'].T[offset:offset+len(feat)]]),-1)
        assert torch.equal(feat,reverse),'pair-feature asymmetry'
        offset+=len(feat)
    h1=model.readout.register_forward_hook(read_hook);h2=model.head.register_forward_pre_hook(check_pair)
    try:raw=model(g)
    finally:h1.remove();h2.remove()
    upper=torch.triu(raw,1);canonical=upper+upper.T+torch.diag_embed(raw.diag())
    idx=torch.triu_indices(len(raw),len(raw),1,device=raw.device)
    assert torch.equal(raw[idx[0],idx[1]],canonical[idx[0],idx[1]])
    assert torch.equal(canonical,canonical.T)
    return canonical

@torch.no_grad()
def evaluate(model,rr,cache,task,out):
    model.eval();zs=[];blocks=[];yy=[];ss=[]
    for i,r in enumerate(rr):
        item=cache.get(r);z=eval_forward(model,move(item['g']),task).cpu().numpy()
        if task=='tlp':
            y=item['target'].numpy();obs=item['observed'].numpy().astype(bool);score=torch.from_numpy(z).sigmoid().numpy();mask=np.triu(~obs,1)
            assert np.max(abs(z-z.T))<1e-5
            yv=y[mask]>0;sv=score[mask];ap=float(average_precision_score(yv,sv)) if yv.any() else 0.
            blocks.append(dict(key=r['key'],missing_edge_ap=ap,missing_positives=int(yv.sum()),unobserved_pairs=int(mask.sum())))
            yy.append(yv);ss.append(sv);np.savez_compressed(out/('prediction_'+r['key']+'.npz'),score=score,target=y,observed=obs)
        else:zs.append(z)
        if i%50==0:save(out/'status.json',dict(state='evaluating_dev',processed=i+1,total=len(rr)))
    if task=='tlp':
        audit=[]
        from tools.cpsf_directional.train import average_precision
        for r in rr:
            with np.load(out/('prediction_'+r['key']+'.npz')) as a:
                idx=np.triu_indices(len(a['target']),1);keep=~a['observed'][idx];audit.append(average_precision(a['target'][idx][keep]>0,a['score'][idx][keep]))
        macro=float(np.mean([v['missing_edge_ap'] for v in blocks]));assert abs(macro-float(np.mean(audit)))<1e-12
        return dict(missing_edge_macro_ap=macro,missing_edge_micro_ap=float(average_precision_score(np.concatenate(yy),np.concatenate(ss))),blocks=blocks,universe='all unobserved unordered node pairs',independent_check=True)
    y=np.array([r['target'] for r in rr]);z=np.array(zs);np.savez_compressed(out/'val_predictions.npz',keys=np.array([r['key'] for r in rr]),target=y,logits=z)
    metrics=calc(y,z,task)
    with np.load(out/'val_predictions.npz') as a:assert metrics==calc(a['target'],a['logits'],task)
    return metrics

def run(task):
    setup();p=verify();cfg=p['tasks'][task];out=ROOT/'results'/('r28_lpa_attention_'+task);out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    if (out/'metrics.json').exists():return
    assert sha(ROOT/'data'/task/'manifest.json')==p['input_manifests'][task]
    rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];assert (len(tr),len(va))==(cfg['train'],cfg['val'])
    assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
    save(out/'status.json',dict(state='verifying_inputs',records=len(rr)))
    for r in rr:
        assert sha(r['path'])==r['sha256'];assert sha(r['feature_path'])==r['feature_sha256']
    m=Model(task).cuda();initial=state_sha(m)
    identity=dict(version=ROOT.name,architecture_version='CPSF_RAW_POINT_ATTENTION_20261019_R28_A1',task=task,kind='r28_lpa_attention',seed=42,protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'data'/task/'manifest.json'),initial_state_sha256=initial,pretrained=False,test_used=False,selection='fixed final epoch',adapter=p['architecture_contract'],runtime_policy_sha256=sha(ROOT/'tools/cpsf_utonia_context/kernel_policy.py'))
    save(out/'identity.json',identity);opt,sched=optimizer(m,task,cfg);start=time.time();steps=0;cache=Cache();history=[];active=set();seen=set();start_epoch=1
    if (out/'latest.pt').exists():
        c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert c['identity']==identity;m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);history=c['history'];steps=c['steps'];start_epoch=c['epoch']+1;np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);seen=set(c['seen'])
    else:torch.save(m.state_dict(),out/'initial.pt')
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,cfg['epochs']+1):
        m.train();begin=time.time();lr=opt.param_groups[0]['lr']
        if task=='ted':
            blocks=sorted({r['source']['block'] for r in tr});lookup={b:[] for b in blocks}
            for r in tr:lookup[r['source']['block']].append(r)
            rng=np.random.RandomState(42+epoch);order=[r for i in rng.permutation(len(blocks)) for r in lookup[blocks[i]]]
        elif task=='tlp':order=[tr[i] for i in np.random.RandomState(42+epoch).permutation(len(tr))]
        else:order=[tr[i] for i in np.random.permutation(len(tr))[:cfg['round_cells']]]
        total=0.
        for lo in range(0,len(order),cfg['batch']):
            group=order[lo:lo+cfg['batch']];opt.zero_grad(set_to_none=True)
            for r in group:
                item=cache.get(r);z=m(move(item['g']));loss=loss_for(z,item,r,task);objective=loss/len(group) if task=='flywire_coarse' else loss
                assert torch.isfinite(loss);objective.backward();total+=float(loss.detach());seen.add(r['key'])
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            if lo==0:active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            if task=='flywire_coarse':torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
            opt.step();steps+=1
            if steps%5==1:save(out/'status.json',dict(state='training',epoch=epoch,epochs=cfg['epochs'],processed=lo+len(group),total=len(order),steps=steps,elapsed_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        sched.step();rec=dict(state_sha256=state_sha(m),epoch=epoch,lr=lr,loss=total/len(order),seconds=time.time()-begin,steps=steps,sample_order_sha256=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest());history.append(rec);save(out/'history.json',history);print(json.dumps(rec),flush=True)
        tmp=out/'latest.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,steps=steps,history=history,identity=identity,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seen=sorted(seen)),tmp);tmp.replace(out/'latest.pt')
    assert steps==cfg['updates'];torch.save(dict(model=m.state_dict(),identity=identity),out/'final.pt');metrics=evaluate(m,va,cache,task,out)
    save(out/'metrics.json',dict(identity=identity,val=metrics,epochs=cfg['epochs'],steps=steps,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),final_sha256=sha(out/'final.pt'),active_gradient_parameters=sorted(active),unique_train_samples_seen=len(seen),test_used=False));save(out/'status.json',dict(state='complete',test_used=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True,choices=['ted','tlp','flywire_coarse']);a=a.parse_args()
    try:run(a.task)
    except BaseException as e:save(ROOT/'results'/('r28_lpa_attention_'+a.task)/'failure.json',dict(error=repr(e)));raise

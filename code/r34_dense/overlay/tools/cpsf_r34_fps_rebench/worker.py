import argparse,fcntl,hashlib,json,math,os,random,time,traceback,gc
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score,average_precision_score,confusion_matrix
from .common import ROOT,read,save,sha,rows,verify
from .backend import Backend,digest,loss_for

def state_steps(opt):
    values=[int(v['step']) for v in opt.state.values() if 'step' in v]
    return dict(min=min(values) if values else 0,max=max(values) if values else 0)

def order_for(tr,task,epoch,cfg,rng):
    if task=='ted':
        blocks=sorted({r['key'].rsplit(':',1)[0] for r in tr});by={b:[] for b in blocks}
        for r in tr:by[r['key'].rsplit(':',1)[0]].append(r)
        return [r for i in np.random.RandomState(42+epoch).permutation(len(blocks)) for r in by[blocks[i]]]
    indices=(np.random.RandomState(42+epoch) if task=='tlp' else rng).permutation(len(tr))
    return [tr[i] for i in indices[:cfg.get('round_cells',len(tr))]]

def optimizer(b,cfg):
    if b.task in ('ted','tlp'):
        opt=torch.optim.Adam(b.model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:1/(e+1)) if b.task=='ted' else torch.optim.lr_scheduler.StepLR(opt,20,.1)
    else:
        opt=torch.optim.AdamW(b.model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        if b.family in ('phase3','utonia'):
            ratio=cfg['lr_min']/cfg['lr'];sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:ratio+(1-ratio)*.5*(1+math.cos(math.pi*e/cfg['epochs'])))
        else:sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr_min'])
    return opt,sched

def acceptance(name,task):
    (ROOT/'acceptance').mkdir(exist_ok=True)
    lock=open(ROOT/'acceptance'/(name+'_'+task+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();rr=rows(task);tr=[r for r in rr if r['split']=='train'];cfg=p['tasks'][task]
    gate=read(ROOT/'input_audits'/(task+'.json'));assert gate['passed']
    assert gate['manifest_sha256']==sha(ROOT/'data'/task/'manifest.json')
    from .contracts import compatibility
    compatibility(task,tr,cfg)
    b=Backend(name,task,tr);m=b.model;initial=digest(m);opt,_=optimizer(b,cfg)
    dims=['nodes','points','patches','regions','incidences'] if name in ('r28','r34') else ['nodes'] if name in ('sgformer','polynormer','graphformer','tednet','tlpnet') else ['points']
    chosen=[min(tr,key=lambda r:r['points'])]+[max(tr,key=lambda r:r[k]) for k in dims]
    chosen=list({r['key']:r for r in chosen}.values());torch.cuda.reset_peak_memory_stats();cases=[];active=set()
    for r in chosen:
        m.train();opt.zero_grad(set_to_none=True);g,y,o=b.item(r);z=m(g);loss=loss_for(z,y,task,b.scale)
        assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        cases.append(dict(key=r['key'],loss=float(loss.detach()),nodes=r['nodes'],points=r['points']))
        del g,y,o,z,loss;gc.collect();torch.cuda.empty_cache()
    if name=='polynormer':
        b.set_epoch(cfg['epochs'],cfg['epochs']);m.train();m.zero_grad(set_to_none=True)
        g,y,o=b.item(max(tr,key=lambda r:r['nodes']));loss=loss_for(m(g),y,task,b.scale);loss.backward()
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        assert any('global' in n for n in active)
        del g,y,o,loss
    required=['native_point.encoder','graph','point','joint','raw_refine.output'] if name in ('r28','r34') else ['native.tree_encoder','native.image_encoder'] if name=='graphformer' else ['backbone'] if b.family in ('phase2','phase3','utonia') else []
    for prefix in required:assert any(n.startswith(prefix) for n in active),('inactive core',name,prefix)
    # Maximum development shape: forward-only, no target enters the model.
    m.eval()
    with torch.no_grad():
        for r in {r['key']:r for r in [max(rr,key=lambda r:r[k]) for k in dims]}.values():
            g,_,_=b.item(r);z=b.eval_forward(g);assert torch.isfinite(z).all();del g,z
    b.set_epoch(1,cfg['epochs']);m.eval();before=digest(m)
    with torch.no_grad():
        g,_,_=b.item(chosen[0]);torch.manual_seed(42);a=b.eval_forward(g).detach().cpu()
        g,_,_=b.item(chosen[-1]);b.eval_forward(g)
        g,_,_=b.item(chosen[0]);torch.manual_seed(42);c=b.eval_forward(g).detach().cpu();del g
    assert torch.allclose(a,c,atol=1e-4,rtol=1e-4),('input independence',float((a-c).abs().max()))
    assert digest(m)==before,'evaluation mutates parameters or buffers'
    trace=[];m.train()
    for r in (chosen[0],chosen[-1]):
        opt.zero_grad(set_to_none=True);g,y,o=b.item(r);loss=loss_for(m(g),y,task,b.scale);loss.backward()
        if task not in ('ted','tlp'):torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        opt.step();trace.append(dict(key=r['key'],loss=float(loss.detach()),optimizer_steps=state_steps(opt)))
    assert state_steps(opt)['max']==2 and digest(m)!=initial
    assert all('region_message.'+n in active for n,_ in m.region_message.named_parameters()), ('inactive R34 parameters', sorted(active))
    report=dict(passed=True,model=name,task=task,protocol_sha256=sha(ROOT/'protocol.json'),
        manifest_sha256=sha(ROOT/'data'/task/'manifest.json'),initial_state_sha256=initial,
        cases=cases,two_updates=trace,active_parameters=sorted(active),official_sources=b.sources,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),input_isolation_max_abs=float((a-c).abs().max()),
        eval_state_unchanged=True,pretrained=False,training_weights_reused=False)
    save(ROOT/'acceptance'/(name+'_'+task+'.json'),report)

def metrics(y,z,task):
    if task=='h01_global':
        return dict(mae=float(np.abs(z-y).mean()),rmse=float(np.sqrt(((z-y)**2).mean())),r2=float(1-((z-y)**2).sum()/((y-y.mean())**2).sum()))
    pred=z.argmax(1)
    if task=='ted':
        tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
        return dict(f1=float(f1_score(y,pred,zero_division=0)),ap=float(average_precision_score(y,z[:,1])),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),decision='argmax two logits for every method')
    if task=='flywire_fine':
        from .metrics import classification
        return classification(y,z)
    labels=list(range(6 if task=='flywire_coarse' else 3))
    return dict(macro_f1=float(f1_score(y,pred,average='macro',labels=labels,zero_division=0)),accuracy=float(accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist())

@torch.no_grad()
def evaluate(b,va,out):
    b.model.eval();before=digest(b.model);zs=[];ys=[];blocks=[];files={}
    for i,r in enumerate(va):
        # Existing point baselines use deterministic per-input evaluation RNG.
        if b.name not in ('r28','r34'):
            seed=int(hashlib.sha256(('42|'+r['key']).encode()).hexdigest()[:8],16);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        g,y,o=b.item(r);z=b.eval_forward(g);assert torch.isfinite(z).all()
        if b.task=='tlp':
            score=z.sigmoid().cpu().numpy();target=y.cpu().numpy();obs=o.cpu().numpy().astype(bool)
            assert np.max(abs(score-score.T))<2e-5
            f=out/'val'/(r['key']+'.npz');f.parent.mkdir(exist_ok=True);np.savez_compressed(f,score=score,target=target,observed=obs);files[str(f)]=sha(f)
        else:ys.append(y);zs.append(z.cpu().numpy())
        if i%50==0:save(out/'status.json',dict(state='evaluating_development',processed=i+1,total=len(va),pid=os.getpid()))
    assert digest(b.model)==before
    if b.task=='tlp':
        missing_y=[];missing_s=[];full_y=[];full_s=[]
        ap=lambda y,s:float(average_precision_score(y,s)) if y.any() else 0.
        for f in files:
            with np.load(f) as a:
                mask=np.triu(~a['observed'],1);off=~np.eye(len(a['target']),dtype=bool)
                y=a['target'][mask];s=a['score'][mask];yf=a['target'][off];sf=a['score'][off]
                blocks.append(dict(key=__import__('pathlib').Path(f).stem,missing_edge_ap=ap(y,s),full_adjacency_ap=ap(yf,sf)))
                missing_y.append(y);missing_s.append(s);full_y.append(yf);full_s.append(sf)
        result=dict(missing_edge_macro_ap=float(np.mean([x['missing_edge_ap'] for x in blocks])),full_adjacency_macro_ap=float(np.mean([x['full_adjacency_ap'] for x in blocks])),
            missing_edge_micro_ap=ap(np.concatenate(missing_y),np.concatenate(missing_s)),full_adjacency_micro_ap=ap(np.concatenate(full_y),np.concatenate(full_s)),blocks=blocks)
    else:
        y=np.asarray(ys);z=np.asarray(zs);result=metrics(y,z,b.task)
        f=out/'val_predictions.npz';np.savez_compressed(f,keys=[r['key'] for r in va],target=y,logits=z);files[str(f)]=sha(f)
        with np.load(f) as a:assert result==metrics(a['target'],a['logits'],b.task)
    save(out/'prediction_audit.json',dict(passed=True,files=files,readback=True,test_used=False))
    return result

def train(name,task):
    p=verify();assert name=='r34';cfg=p['tasks'][task];rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
    gate=read(ROOT/'acceptance'/(name+'_'+task+'.json'));assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    assert gate['manifest_sha256']==sha(ROOT/'data'/task/'manifest.json')
    out=ROOT/'results'/(name+'_'+task);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'metrics.json').exists():return
    b=Backend(name,task,tr);m=b.model;initial=digest(m);assert initial==gate['initial_state_sha256']
    identity=dict(version=ROOT.name,model=name,task=task,seed=42,protocol_sha256=sha(ROOT/'protocol.json'),
        manifest_sha256=gate['manifest_sha256'],initial_state_sha256=initial,official_sources=b.sources,
        pretrained=False,test_used=False,selection='fixed final epoch')
    save(out/'identity.json',identity);opt,sched=optimizer(b,cfg);rng=np.random.RandomState(42)
    history=[];steps=0;start_epoch=1;start=time.time()
    if (out/'latest.pt').exists():
        c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);assert c['identity']==identity
        m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);history=c['history'];steps=c['steps'];start_epoch=c['epoch']+1
        rng.set_state(c['order_rng']);random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);del c
    else:np.random.seed(42)
    torch.cuda.reset_peak_memory_stats();orders=read(ROOT/'orders'/f'{task}.json')
    for epoch in range(start_epoch,cfg['epochs']+1):
        m.train();b.set_epoch(epoch,cfg['epochs']);order=order_for(tr,task,epoch,cfg,rng)
        order_hash=hashlib.sha256('\n'.join(r['key'] for r in order).encode()).hexdigest();assert order_hash==orders[epoch-1]['sha256']
        total=0.;begin=time.time();lr=opt.param_groups[0]['lr']
        for lo in range(0,len(order),cfg['batch']):
            group=order[lo:lo+cfg['batch']];opt.zero_grad(set_to_none=True)
            for r in group:
                g,y,o=b.item(r);z=m(g);loss=loss_for(z,y,task,b.scale);assert torch.isfinite(loss),(r['key'],loss)
                (loss if task in ('ted','tlp') else loss/len(group)).backward();total+=float(loss.detach());del g,y,o,z,loss
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            if task not in ('ted','tlp'):torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
            opt.step();steps+=1
            if steps==1:save(out/'first_update.json',dict(initial=initial,after=digest(m),optimizer_steps=state_steps(opt),passed=digest(m)!=initial))
            if steps%5==1:save(out/'status.json',dict(state='training',epoch=epoch,epochs=cfg['epochs'],steps=steps,
                optimizer_steps=state_steps(opt),processed=lo+len(group),total=len(order),loss=total/(lo+len(group)),
                seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        sched.step();history.append(dict(epoch=epoch,steps=steps,loss=total/len(order),lr=lr,seconds=time.time()-begin,sample_order_sha256=order_hash));save(out/'history.json',history)
        temp=out/'latest.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),identity=identity,
            epoch=epoch,steps=steps,history=history,order_rng=rng.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(out/'latest.pt')
        print(json.dumps(history[-1]),flush=True)
    assert steps==cfg['updates'];torch.save(dict(model=m.state_dict(),identity=identity,epoch=cfg['epochs']),out/'final.pt')
    result=evaluate(b,va,out);verify()
    save(out/'metrics.json',dict(identity=identity,val=result,epochs=cfg['epochs'],steps=steps,final_sha256=sha(out/'final.pt'),
        seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),test_used=False))
    save(out/'status.json',dict(state='complete',steps=steps,epoch=cfg['epochs']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--task',required=True);a.add_argument('--accept',action='store_true');a=a.parse_args()
    try:acceptance(a.model,a.task) if a.accept else train(a.model,a.task)
    except BaseException as e:
        save(ROOT/('acceptance' if a.accept else 'results')/(a.model+'_'+a.task+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise

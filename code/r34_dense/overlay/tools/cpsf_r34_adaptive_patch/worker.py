import argparse,gc,math,os,time,traceback,fcntl
import numpy as np
import torch
from .common import ROOT,REFERENCE,rows,read,save,sha,verify
from .backend import Backend,digest,loss_for
from .data import parent_load
from tools.cpsf_r34_fps_rebench import worker as inherited
from tools.cpsf_r34_fps_rebench.backend import Backend as ParentBackend


def compatibility(task,tr,cfg):
    sample=min(tr,key=lambda r:r['incidences']);records=[]
    for original in [True,False]:
        b=ParentBackend('r34',task,tr) if original else Backend('r34',task,tr,force_parent=True)
        m=b.model;initial=digest(m);opt,_=inherited.optimizer(b,cfg);outputs=[];states=[]
        for _ in range(2):
            m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(sample);z=m(g)
            loss=loss_for(z,y,task,b.scale);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
            outputs.append(z.detach().cpu());states.append(digest(m))
        records.append(dict(initial=initial,outputs=outputs,states=states))
        del b,m,opt,g,y,z,loss;gc.collect();torch.cuda.empty_cache()
    assert records[0]['initial']==records[1]['initial']
    assert records[0]['states']==records[1]['states'],'Fixed-size control update drift'
    assert all(torch.equal(a,b) for a,b in zip(records[0]['outputs'],records[1]['outputs']))
    return dict(passed=True,key=sample['key'],two_update_states=records[0]['states'],initial=records[0]['initial'])


def acceptance(task):
    folder=ROOT/'acceptance';folder.mkdir(exist_ok=True)
    lock=open(folder/('r34_'+task+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();cfg=p['tasks'][task];rr=rows(task);tr=[r for r in rr if r['split']=='train']
    state_path=folder/(task+'_progress.json');save(state_path,dict(state='fixed_size_control',pid=os.getpid()))
    control=compatibility(task,tr,cfg)
    save(folder/(task+'_compatibility.json'),control)
    b=Backend('r34',task,tr);m=b.model;initial=digest(m);assert initial==control['initial']
    torch.cuda.reset_peak_memory_stats();active=set();cases=[]
    dims=['nodes','points','patches','regions','incidences']
    chosen=[min(tr,key=lambda r:r['points'])]+[max(tr,key=lambda r:r[k]) for k in dims]
    pinned='883843993' if task=='h01_global' else '720575940608618197'
    chosen += [r for r in tr if r['key']==pinned]
    chosen=list({r['key']:r for r in chosen}.values())
    for row in chosen:
        save(state_path,dict(state='extreme_train_backward',key=row['key'],processed=len(cases),total=len(chosen),pid=os.getpid()))
        m.train();m.zero_grad(set_to_none=True);g,y,_=b.item(row);seen=[]
        parent=parent_load(row)['g']
        for field in ['ucoord','ugrid','uinverse']:
            assert torch.equal(g[field].cpu(),parent[field]),('context geometry drift',field)
        del parent
        handle=m.scale.register_forward_hook(lambda module,args,value:seen.append(args[0].detach().clone()))
        try:z=m(g);loss=loss_for(z,y,task,b.scale);assert torch.isfinite(loss);loss.backward()
        finally:handle.remove()
        assert len(seen)==1 and torch.allclose(seen[0][:,2],torch.full_like(seen[0][:,2],math.log1p(g['patch_size'])))
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        cases.append(dict(key=row['key'],loss=float(loss.detach()),**b.cache.get(row)['audit']))
        del g,y,z,loss,seen;gc.collect();torch.cuda.empty_cache()
    m.eval()
    with torch.no_grad():
        for row in {r['key']:r for r in [max(rr,key=lambda r:r[k]) for k in dims]}.values():
            save(state_path,dict(state='extreme_development_shape_forward',key=row['key'],pid=os.getpid()))
            g,_,_=b.item(row);assert torch.isfinite(m(g)).all();del g
    m.eval();before=digest(m)
    with torch.no_grad():
        g,_,_=b.item(chosen[0]);a=m(g).detach().cpu()
        g,_,_=b.item(chosen[-1]);m(g)
        g,_,_=b.item(chosen[0]);c=m(g).detach().cpu();del g
    assert torch.equal(a,c) and digest(m)==before
    opt,_=inherited.optimizer(b,cfg);trace=[];m.train()
    for row in [chosen[0],chosen[-1]]:
        opt.zero_grad(set_to_none=True);g,y,_=b.item(row);loss=loss_for(m(g),y,task,b.scale);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(key=row['key'],loss=float(loss.detach()),optimizer_steps=inherited.state_steps(opt)))
    assert inherited.state_steps(opt)['max']==2 and digest(m)!=initial
    for prefix in ['native_point.encoder','graph','point','joint','raw_refine.output','scale']:
        assert any(n.startswith(prefix) for n in active),('inactive core',prefix)
    assert all('region_message.'+n in active for n,_ in m.region_message.named_parameters())
    result=dict(passed=True,model='r34',task=task,protocol_sha256=sha(ROOT/'protocol.json'),
         manifest_sha256=p['input_manifests'][task],initial_state_sha256=initial,official_sources=b.sources,
         compatibility=control,cases=cases,peak_gpu_bytes=torch.cuda.max_memory_allocated(),two_updates=trace,
         actual_scale_verified=True,input_isolation_exact=True,eval_state_unchanged=True,
         utonia_context_grid_and_coordinates_unchanged=True,
         active_parameters=sorted(active),pretrained=False,training_weights_reused=False)
    save(folder/('r34_'+task+'.json'),result);save(state_path,dict(state='passed',pid=os.getpid(),peak_gpu_bytes=result['peak_gpu_bytes']))
    print('ACCEPTED',task,result['peak_gpu_bytes'],flush=True)


def main(task,accept):
    if accept:return acceptance(task)
    # Reuse the exact frozen training/evaluation loop; redirect only its explicit
    # experiment namespace and backend. No edits to the reference worker file.
    inherited.ROOT=ROOT;inherited.rows=rows;inherited.verify=verify;inherited.Backend=Backend
    inherited.train('r34',task)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True,choices=['h01_global','flywire_coarse']);a.add_argument('--accept',action='store_true');a=a.parse_args()
    try:main(a.task,a.accept)
    except BaseException as e:
        save(ROOT/('acceptance' if a.accept else 'results')/('r34_'+a.task+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise

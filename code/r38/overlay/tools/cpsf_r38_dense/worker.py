import argparse,fcntl,gc,os,traceback
import torch
from .common import ROOT,R38,AP,TASKS,rows,read,save,sha,verify
from .backend import Backend,digest,loss_for
from .data import parent_load,DENSE_FIELDS
from tools.cpsf_cmm_pair_fusion import worker as inherited
from tools.cpsf_cmm_pair_fusion.backend import Backend as R38Backend
from tools.cpsf_r34_fps_rebench.backend import Backend as R34Backend
from tools.cpsf_cmm_pair_fusion.acceptance import fusion_interface_contract,checkpoint_contract

def compatibility(task,tr,cfg):
    row=min(tr,key=lambda r:r['points'])
    old=R38Backend('r38',task,tr) if task=='h01_global' else R34Backend('r34',task,tr)
    new=Backend('r38',task,tr,force_parent=True);initial=digest(new.model)
    if task=='flywire_coarse':new.model.cmm_pair_fusion.enabled=False
    shared=old.model.state_dict()
    assert all(torch.equal(v,new.model.state_dict()[n]) for n,v in shared.items())
    g,y,_=new.item(row);old.model.eval();new.model.eval()
    with torch.no_grad():assert torch.equal(old.model(g),new.model(g))
    opt0,_=inherited.optimizer(old,cfg);opt1,_=inherited.optimizer(new,cfg);trace=[]
    for step in range(2):
        values=[]
        for b,opt in ((old,opt0),(new,opt1)):
            torch.manual_seed(42+step);b.model.train();opt.zero_grad(set_to_none=True)
            z=b.model(g);loss=loss_for(z,y,task,b.scale);loss.backward()
            torch.nn.utils.clip_grad_norm_(b.model.parameters(),5,error_if_nonfinite=True);opt.step();values.append(float(loss.detach()))
        assert values[0]==values[1]
        assert all(torch.equal(v,new.model.state_dict()[n]) for n,v in old.model.state_dict().items())
        trace.append({'step':step+1,'loss':values[0],'shared_parameters_exact':True})
    result=dict(passed=True,initial=initial,key=row['key'],reference='original R38' if task=='h01_global' else 'original R34 with pair fusion disabled',two_steps=trace)
    del old,new,opt0,opt1,g,y,z,loss,shared;gc.collect();torch.cuda.empty_cache()
    return result

def acceptance(task):
    folder=ROOT/'acceptance';folder.mkdir(exist_ok=True)
    lock=open(folder/('r38_'+task+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();cfg=p['tasks'][task];rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
    progress=folder/(task+'_progress.json');save(progress,dict(state='compatibility',pid=os.getpid()))
    control=compatibility(task,tr,cfg);save(folder/(task+'_compatibility.json'),control)
    b=Backend('r38',task,tr);m=b.model;initial=digest(m);assert initial==control['initial']
    opt,_=inherited.optimizer(b,cfg);active=set();trace=[];torch.cuda.reset_peak_memory_stats()
    warm=min(tr,key=lambda r:r['points'])
    for step in range(6):
        m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(warm);loss=loss_for(m(g),y,task,b.scale);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(step=step+1,loss=float(loss.detach()),optimizer_steps=inherited.state_steps(opt)))
    assert {n for n,_ in m.named_parameters() if n.startswith('cmm_pair_fusion.')}<=active
    fusion=fusion_interface_contract(b,warm);check=checkpoint_contract(b,warm)
    dims=['nodes','points','patches','regions','incidences'];chosen=[warm]+[max(tr,key=lambda r:r[k]) for k in dims]
    # Include extrema measured in the existing AP cache when available. Other
    # samples are generated directly from the same deterministic rule on read.
    audits={r['key']:read(AP/'adaptive_cache'/task/(r['key']+'.json'))['audit'] for r in tr if (AP/'adaptive_cache'/task/(r['key']+'.json')).exists()}
    if audits:
        chosen.extend(next(r for r in tr if r['key']==key) for key in [max(audits,key=lambda k:audits[k]['after']['maximum']),max(audits,key=lambda k:audits[k]['patch_size'])])
    cases=[]
    for row in {r['key']:r for r in chosen}.values():
        save(progress,dict(state='maximum_train_backward',key=row['key'],completed=len(cases),pid=os.getpid()))
        m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(row);original=parent_load(row)['g']
        for k,v in original.items():
            if isinstance(v,torch.Tensor):assert torch.equal(g[k].cpu(),v),k
        # Check against the successful R34 dense interface, not only our own inverse.
        path=AP/'adaptive_cache'/task/(row['key']+'.pt')
        if path.exists():
            assert sha(path)==read(path.with_suffix('.json'))['cache_sha256'];ap=torch.load(path,map_location='cpu',weights_only=False)['g']
            if 'dense_size' in g:
                for k in DENSE_FIELDS:
                    if isinstance(ap[k],torch.Tensor):assert torch.equal(g['dense_'+k].cpu(),ap[k]),k
                    else:assert g['dense_'+k]==ap[k],k
                assert torch.equal(g['dense_pf'].cpu(),ap['pf']*(ap['patch_size']/.5))
            del ap
        del original
        z=m(g);loss=loss_for(z,y,task,b.scale);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        cases.append(dict(key=row['key'],loss=float(loss.detach()),**b.cache.get(row)['audit']))
        del g,y,z,loss;gc.collect();torch.cuda.empty_cache()
    for prefix in ['native_point.encoder','graph','point','joint','raw_refine.output','region_message','cmm_pair_fusion']:
        assert any(n.startswith(prefix) for n in active),prefix
    m.eval();before=digest(m);dev=[]
    with torch.no_grad():
        for row in {r['key']:r for r in [max(va,key=lambda r:r[k]) for k in dims]}.values():
            g,_,_=b.item(row);assert torch.isfinite(b.eval_forward(g)).all();dev.append(row['key']);del g
        g,_,_=b.item(warm);a=b.eval_forward(g).cpu()
        g,_,_=b.item(chosen[-1]);b.eval_forward(g)
        g,_,_=b.item(warm);c=b.eval_forward(g).cpu()
    assert torch.equal(a,c) and digest(m)==before
    result=dict(passed=True,model='r38',task=task,protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=p['input_manifests'][task],initial_state_sha256=initial,
        official_sources=b.sources,compatibility=control,updates=trace,cases=cases,development_shape_cases=dev,fusion_contract=fusion,checkpoint_contract=check,
        original_fields_exact=True,utonia_input_exact=True,dense_interface_matches_r34=True,active_parameters=sorted(active),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),input_isolation_exact=True,eval_state_unchanged=True,test_used=False,pretrained=False,training_weights_reused=False)
    save(folder/('r38_'+task+'.json'),result);save(progress,dict(state='passed',peak_gpu_bytes=result['peak_gpu_bytes'],pid=os.getpid()))
    print('ACCEPTED',task,result['peak_gpu_bytes'],flush=True)

def main(task,accept):
    if accept:return acceptance(task)
    inherited.ROOT=ROOT;inherited.rows=rows;inherited.verify=verify;inherited.Backend=Backend
    inherited.train('r38',task)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True,choices=TASKS);parser.add_argument('--accept',action='store_true');a=parser.parse_args()
    try:main(a.task,a.accept)
    except BaseException as e:
        save(ROOT/('acceptance' if a.accept else 'results')/('r38_'+a.task+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise

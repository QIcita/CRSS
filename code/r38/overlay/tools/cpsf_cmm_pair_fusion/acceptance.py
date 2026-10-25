"""Real H01 acceptance without using held-out outcomes for model selection."""
import fcntl,gc,hashlib,time
import torch
from .common import ROOT,PARENT,PREFIX,read,save,sha,rows,verify
from .backend import Backend,digest,loss_for
from .operator import gates,set_enabled


def fusion_interface_contract(backend,row):
    from tools.cpsf_v21_segment_units.model import add
    m=backend.model;g,_,_=backend.item(row);m.eval();seen={};counts={'utonia':0,'spotr':0};handles=[]
    original=m.patch_context_delta
    def wrap(h,p):
        value=original(h,p);seen['patch']=h+value;return value
    def count(key):
        def hook(module,args,value):counts[key]+=1
        return hook
    def capture(key):
        def hook(module,args,value):seen[key]=value
        return hook
    def fusion_inputs(module,args):seen['inputs']=args
    def message_inputs(module,args):seen['message_patch']=args[1]
    def surface_inputs(module,args):seen['surface_patch']=m._message_patch
    m.patch_context_delta=wrap
    handles.extend([m.native_point.register_forward_hook(count('utonia')),
        m.raw_refine.register_forward_hook(count('spotr')),
        m.cmm_pair_fusion.register_forward_pre_hook(fusion_inputs),
        m.region_message.register_forward_pre_hook(message_inputs),
        m.relation.register_forward_pre_hook(surface_inputs)])
    for key,module in [('relation',m.relation),('position',m.position),('scale',m.scale),('message',m.region_message)]:
        handles.append(module.register_forward_hook(capture(key)))
    try:
        with torch.no_grad():
            _,point=m.stems(g)
            assert counts=={'utonia':1,'spotr':1},counts
            assert torch.equal(seen['patch'],seen['message_patch']) and torch.equal(seen['patch'],seen['surface_patch'])
            wp=g['pw']*g['patch_mass'][g['pj']]
            surface=add((seen['patch'][g['pj']]+seen['relation'])*wp[:,None],g['pr'],g['nregions'])/g['surface_mass'].clamp_min(1e-12)[:,None]
            expected=(surface+(seen['position']+seen['scale']))*g['has_p'][:,None]
            expected=expected+.1*(seen['message']*(g['has_g']&g['has_p'])[:,None])
            assert torch.equal(point,expected),float((point-expected).abs().max())
            terms=m.cmm_pair_fusion.terms(*seen['inputs'])
            pair_max=[float(t.abs().max()) for t in terms]
            assert all(v>0 for v in pair_max),pair_max
            bound=.1*sum(w*x.norm(dim=-1) for x,w in zip(seen['inputs'],(1.,.1,.1)))
            ratio=sum(terms).norm(dim=-1)/bound.clamp_min(1e-12)
            assert bool((ratio<=1+2e-6).all())
    finally:
        for handle in handles:handle.remove()
        del m.patch_context_delta
    m.train();params=[v for v in m.parameters() if v.requires_grad];m.cmm_pair_fusion.checkpointed=True
    torch.manual_seed(4242);a=m(g);ga=torch.autograd.grad(a.sum(),params,allow_unused=True)
    m.cmm_pair_fusion.checkpointed=False
    torch.manual_seed(4242);b=m(g);gb=torch.autograd.grad(b.sum(),params,allow_unused=True)
    m.cmm_pair_fusion.checkpointed=True
    assert torch.equal(a,b);errors=[]
    for x,y in zip(ga,gb):
        assert (x is None)==(y is None)
        if x is not None:
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5);errors.append(float((x-y).abs().max()))
    return dict(passed=True,one_call_per_backbone=True,counts=counts,shared_patch_exact=True,
        surface_reconstruction_exact=True,pair_delta_max_abs=pair_max,bound_max_ratio=float(ratio.max()),
        full_model_fusion_checkpoint_output_exact=True,gradient_max_abs=max(errors,default=0.))


def compatibility(task,tr,cfg):
    from .worker import optimizer
    row=min(tr,key=lambda r:r['points'])
    old=Backend('r34',task,tr);new=Backend('r38',task,tr)
    shared={n:v for n,v in new.model.state_dict().items() if PREFIX not in n}
    assert shared.keys()==old.model.state_dict().keys()
    assert all(torch.equal(v,old.model.state_dict()[n]) for n,v in shared.items())
    g,y,_=new.item(row);old.model.eval();new.model.eval()
    with torch.no_grad():
        a=old.model(g);b=new.model(g);assert torch.equal(a,b),float((a-b).abs().max())
    set_enabled(new.model,False)
    opt0,_=optimizer(old,cfg);opt1,_=optimizer(new,cfg);trace=[]
    for step in range(2):
        values=[]
        for backend,opt in ((old,opt0),(new,opt1)):
            torch.manual_seed(42+step);backend.model.train();opt.zero_grad(set_to_none=True)
            loss=loss_for(backend.model(g),y,task,backend.scale);loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.model.parameters(),5,error_if_nonfinite=True)
            opt.step();values.append(float(loss.detach()))
        assert values[0]==values[1]
        assert all(torch.equal(v,new.model.state_dict()[n]) for n,v in old.model.state_dict().items())
        trace.append(dict(step=step+1,loss=values[0],shared_parameters_exact=True))
    save(ROOT/'contracts'/(task+'.json'),dict(passed=True,task=task,sample=row['key'],
         original_initialization_exact=True,zero_init_exact=True,disabled_two_updates=trace,
         parent_protocol_sha256=sha(PARENT/'protocol.json'),test_used=False))
    del old,new,opt0,opt1,g,y,loss,shared,a,b;gc.collect();torch.cuda.empty_cache()


def checkpoint_contract(backend,row):
    g,_,_=backend.item(row)
    ref=backend.model.native_point
    wrappers=[m for m in ref.modules() if type(m).__name__=='NativeCheckpoint']
    assert len(wrappers)==4 and all(m.enabled for m in wrappers)
    params=[p for p in ref.parameters() if p.requires_grad]
    target=torch.sin(torch.arange(len(g['pf'])*96,device=g['pf'].device,dtype=g['pf'].dtype)).reshape(len(g['pf']),96)
    torch.manual_seed(4242)
    a=ref(g);ga=torch.autograd.grad((a*target).sum(),params,allow_unused=True)
    for module in wrappers:module.enabled=False
    torch.manual_seed(4242)
    b=ref(g);gb=torch.autograd.grad((b*target).sum(),params,allow_unused=True)
    for module in wrappers:module.enabled=True
    assert torch.equal(a,b)
    errors=[]
    for x,y in zip(ga,gb):
        assert (x is None)==(y is None)
        if x is not None:
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
            errors.append(float((x-y).abs().max()))
    return dict(passed=True,checkpoint_blocks=len(wrappers),prediction_exact=True,gradient_max_abs=max(errors,default=0.))


def acceptance(name,task):
    from .worker import optimizer,state_steps
    assert name=='r38'
    lockpath=ROOT/'acceptance'/(name+'_'+task+'.lock');lockpath.parent.mkdir(exist_ok=True)
    lock=open(lockpath,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();cfg=p['tasks'][task];rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
    gate=read(ROOT/'input_audits'/(task+'.json'));assert gate['passed']
    assert gate['manifest_sha256']==sha(ROOT/'data'/task/'manifest.json')
    compatibility(task,tr,cfg)
    b=Backend(name,task,tr);m=b.model;initial=digest(m);opt,_=optimizer(b,cfg)
    dims=['nodes','points','patches','regions','incidences']
    chosen=list({r['key']:r for r in [min(tr,key=lambda r:r['points'])]+[max(tr,key=lambda r:r[k]) for k in dims]}.values())
    active=set();trace=[];torch.cuda.reset_peak_memory_stats();t0=time.time()
    # Warm-up uses only the minimum training case; nested zero initializations
    # are checked using a few training-only updates; all acceptance weights are discarded.
    for step in range(6):
        m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(chosen[0]);loss=loss_for(m(g),y,task,b.scale);loss.backward()
        assert torch.isfinite(loss)
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(step=step+1,loss=float(loss.detach()),optimizer_steps=state_steps(opt)))
    added=[n for n,_ in m.named_parameters() if PREFIX in n]
    assert set(added)<=active,('inactive added parameters',set(added)-active)
    check=checkpoint_contract(b,chosen[0]);fusion_check=fusion_interface_contract(b,chosen[0]);cases=[]
    for r in chosen:
        m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(r);z=m(g);loss=loss_for(z,y,task,b.scale);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        cases.append(dict(key=r['key'],split='train',loss=float(loss.detach()),**{k:r[k] for k in dims}))
        del g,y,z,loss;gc.collect();torch.cuda.empty_cache()
        save(ROOT/'acceptance'/(name+'_'+task+'_progress.json'),dict(state='maximum_shape_acceptance',cases=cases,pid=__import__('os').getpid()))
    for prefix in ('native_point.encoder','graph','point','joint','raw_refine.output','region_message'):
        assert any(n.startswith(prefix) for n in active),prefix
    dev_cases=[];m.eval();before=digest(m);internal_effect=[]
    def capture_effect(module,args,value):
        internal_effect.append(float(value.abs().max()))
    handles=[gate.register_forward_hook(capture_effect) for gate in gates(m)]
    with torch.no_grad():
        for r in {r['key']:r for r in [max(va,key=lambda r:r[k]) for k in dims]}.values():
            g,_,_=b.item(r);z=b.eval_forward(g);assert torch.isfinite(z).all()
            dev_cases.append(dict(key=r['key'],split='val',target_used=False,**{k:r[k] for k in dims}));del g,z
        g,_,_=b.item(chosen[0]);a=b.eval_forward(g).cpu()
        g,_,_=b.item(chosen[-1]);b.eval_forward(g)
        g,_,_=b.item(chosen[0]);c=b.eval_forward(g).cpu()
        assert torch.equal(a,c)
        set_enabled(m,False);off=b.eval_forward(g).cpu()
        set_enabled(m,True)
    for handle in handles:handle.remove()
    assert digest(m)==before
    assert internal_effect and max(internal_effect)>0
    report=dict(passed=True,model=name,task=task,protocol_sha256=sha(ROOT/'protocol.json'),
        manifest_sha256=gate['manifest_sha256'],initial_state_sha256=initial,added_parameters=9216,
        cases=cases,development_forward_cases=dev_cases,updates=trace,checkpoint_contract=check,fusion_contract=fusion_check,
        active_parameters=sorted(active),official_sources=b.sources,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        eval_state_unchanged=True,input_isolation_exact=True,trained_operator_effect=float((a-off).abs().max()),
        internal_fusion_delta_max_abs=max(internal_effect),
        effect_gate='three nonzero pair terms, complete interface consistency, bounded delta and active gradients; prediction difference is diagnostic',
        pretrained=False,test_used=False,training_weights_reused=False,seconds=time.time()-t0)
    save(ROOT/'acceptance'/(name+'_'+task+'.json'),report)
    print('ACCEPTED',task,report['peak_gpu_bytes'],flush=True)

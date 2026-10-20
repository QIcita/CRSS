import argparse,gc,time,json
import numpy as np
import torch
from .common import ROOT,BASE,setup,verify,sha,save,rows,state_sha
from tools.cpsf_fixed_core.model import FlyModel
from tools.cpsf_directional.model import Model as OldTLP
from tools.cpsf_fixed_nredd.model import Model as OldTED
from .model import Model
from .data import load_item,raw,move
from .train import optimizer,loss_for,eval_forward
from tools.cpsf_r9_expansion.model import Model as R9
from tools.cpsf_raw_point_attention.contracts import check_grouping

def main(task):
    setup();protocol=verify();rr=[r for r in rows(task) if r['split']=='train'];dims=['nodes','regions','patches','points','incidences']
    if task=='tlp':dims+=['query_nodes','bridge_incidences']
    chosen=[min(rr,key=lambda r:r['points']),sorted(rr,key=lambda r:r['points'])[len(rr)//2]]+[max(rr,key=lambda r:r[k]) for k in dims]
    if task=='flywire_coarse':chosen += [next(r for r in rr if r['target']==c) for c in range(6)]
    chosen=list({r['key']:r for r in chosen}.values());report=dict(task=task,protocol_sha256=sha(ROOT/'protocol.json'),cases=[],test_used=False)
    setup();m=Model(task).cuda();setup();base=R9(task).cuda()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in base.state_dict().items())
    report['old_initial_parameters_exact']=True
    for r in chosen:
        start=time.time();item=load_item(r);g=move(item['g']);m.eval();base.eval()
        check_grouping(g);m.raw_refine.output.weight.data.zero_()
        oldg=dict(g)
        # R9 input adapter is the same audited observation and directional features.
        with torch.no_grad():a=m(g);b=base(oldg)
        err=float(abs(a-b).max());assert torch.allclose(a,b,atol=2e-6,rtol=2e-6),(task,r['key'],err)
        m.train();m.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();z=m(g);loss=loss_for(z,item,r,task);assert torch.isfinite(loss);loss.backward()
        active=[n for n,p in m.named_parameters() if p.grad is not None and bool((p.grad!=0).any())]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for prefix in ['native_point.encoder','graph','point','joint','processor','position','scale']:
            assert any(n.startswith(prefix) for n in active),(task,r['key'],prefix)
        eligible=int((g['raw_counts']>1).sum())
        if eligible:assert any(n.startswith('raw_refine.output') for n in active)
        if task=='tlp':
            assert z.shape==item['target'].shape
            m.eval()
            with torch.no_grad():assert torch.isfinite(eval_forward(m,g,task)).all()
            m.train()
        report['cases'].append(dict(key=r['key'],old_core_maxdiff=err,points=len(g['pf']),voxels=len(g['ufeat']),regions=g['nregions'],peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,finite_gradients=True,non_singleton_patches=eligible))
        print(task,r['key'],report['cases'][-1],flush=True)
        del item,g,oldg,z,loss,a,b;gc.collect();torch.cuda.empty_cache()
    del m,base;gc.collect();torch.cuda.empty_cache()
    # Two genuine optimization updates, independently repeatable across GPUs.
    setup();m=Model(task).cuda();opt,_=optimizer(m,task,protocol['tasks'][task]);trace=[]
    small=sorted(rr,key=lambda r:r['points'])[:2]
    for r in small:
        opt.zero_grad(set_to_none=True);item=load_item(r);z=m(move(item['g']));loss=loss_for(z,item,r,task);loss.backward()
        if task=='flywire_coarse':torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
        opt.step();trace.append(dict(key=r['key'],loss=float(loss.detach()),state_sha256=state_sha(m)))
    report.update(passed=True,trace=trace,parameters=sum(v.numel() for v in m.parameters()))
    save(ROOT/'acceptance'/(task+'_rep'+__import__('os').environ.get('CPSF_ACCEPT_REP','1')+'.json'),report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args();main(a.task)

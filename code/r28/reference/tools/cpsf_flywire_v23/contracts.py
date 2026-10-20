import json,time
import numpy as np
import torch
from .data import ROOT,read_dataset,load
from .model import Model,ARMS,variant
from .train import move,dataset
from tools.cpsf_v21_segment_units.io import setup,save
from tools.cpsf_v23_residual_region.model import Model as Baseline
from tools.cpsf_flywire_20260930.evaluation import classify,retrieve

def main():
    rows,labels,ready=read_dataset();parts={s:{r['root_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not parts['train']&parts['val'] and not parts['train']&parts['test'] and not parts['val']&parts['test']
    from collections import Counter
    freq=Counter(r['cell_type'] for r in rows if r['split']=='train');assert min(freq.values())>=3
    manifest=json.loads((ROOT/'profile_cache/manifest.json').read_text())
    records=manifest['records'];assert all(r['split']=='train' for r in records)
    largest=max((r for r in records if r['variant']=='joint'),key=lambda r:r['nodes'])['root_id']
    checks={}
    for task in ('coarse','fine'):
        initial=None
        for arm in ARMS:
            print('CHECK',task,arm,flush=True)
            r=next(r for r in records if r['root_id']==largest and r['variant']==variant(arm));g=move(load(r['path']))
            setup();model=Model(len(labels[task]),arm).cuda();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if initial is None:initial=state
            else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
            if arm=='full':
                baseline=Baseline('global','full').cuda()
                base_state=baseline.state_dict();base_state.update({k:v for k,v in state.items() if not k.startswith('head.')});baseline.load_state_dict(base_state)
                captured=[];hook=baseline.head.register_forward_pre_hook(lambda m,args:captured.append(args[0]))
                with torch.no_grad():
                    baseline(g);torch.testing.assert_close(captured[0],model.embedding(g),rtol=1e-5,atol=1e-6)
                hook.remove();del baseline,captured
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            torch.cuda.reset_peak_memory_stats();start=time.time()
            for step in range(2):
                opt.zero_grad(set_to_none=True);z=model(g)
                loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([r[task+'_target']],device='cuda'),label_smoothing=.05)
                assert torch.isfinite(loss);loss.backward()
                for name,on in [('position',model.use_position),('scale',model.use_scale)]:
                    grad=getattr(model,name)[0].weight.grad
                    assert (grad is not None)==on
                    if on:assert torch.isfinite(grad).all() and grad.abs().sum()>0
                if arm=='point':assert model.graph[0].weight.grad is None
                if arm=='skeleton':assert model.point[0].weight.grad is None
                if arm=='mean_only':assert model.joint[0].weight.grad is None
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();checks[task+'_'+arm]=dict(passed=True,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del model,opt,g,z,loss,state;torch.cuda.empty_cache()
    for rid in {r['root_id'] for r in records}:
        rr={r['variant']:r for r in records if r['root_id']==rid}
        gs={k:load(r['path']) for k,r in rr.items()}
        full=gs['joint']
        for name in ('thumbnail','uniform_budget'):
            assert torch.equal(full['s'],gs[name]['s'])
            assert full['raw_edges']==gs[name]['raw_edges']
            torch.testing.assert_close(full['surface_mass'].sum(),gs[name]['surface_mass'].sum(),rtol=2e-5,atol=.001)
        for field in ('pf','point_weight','point_owner','patch_mass'):
            assert torch.equal(full[field],gs['point'][field]),(rid,field)
        assert rr['joint']['points']==rr['uniform_budget']['points'] and rr['thumbnail']['points']==4096
        assert not gs['point']['has_g'].any() and not gs['skeleton']['has_p'].any()
    assert classify(np.arange(148),np.eye(148),148)['macro_f1']==1
    retrieval=retrieve(np.eye(2),np.repeat(np.eye(2),3,axis=0),['a','b'],['a']*3+['b']*3,['q1','q2'],['g'+str(i) for i in range(6)])
    assert retrieval['cell_type_macro_mAP']==1 and retrieval['hit_rate']['1']==1
    estimate=json.loads((ROOT/'cache_estimate.json').read_text())
    assert estimate['estimated_bytes']<40*2**30,'Estimated cache needs review before full build'
    save(ROOT/'contracts.json',dict(passed=True,training_geometry_only=True,checks=checks,
        largest_training_root=largest,full_embedding_matches_h01_encoder_within_fp32_tolerance=True,
        pure_modality_support=True,complete_skeleton=True,same_surface_joint_point=True,
        split_root_disjoint=True,test_used=False,accuracy_claim=False))

if __name__=='__main__':main()

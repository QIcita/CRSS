import json
import torch
from tools.cpsf_v21_segment_units.io import setup,load,save,sha
from tools.cpsf_v23_residual_region.model import Model as Baseline
from .model import Model,JOBS
from .prepare import ROOT,BASE
from .train import move


def main():
    records=json.loads((ROOT/'smoke_cache/manifest.json').read_text())['records']
    assert all(r['split']=='train' for r in records)
    checks={}
    for task,arm in [('local','full'),('global','full')]+JOBS:
        modal=arm if arm in ('thumbnail','uniform_budget') else 'joint'
        row=next(r for r in records if r['task']==task and r['modal']==modal)
        assert sha(row['path'])==row['sha256'];g=move(load(row['path']))
        setup();baseline=Baseline(task,'full').cuda();rng=torch.get_rng_state()
        reference={k:v.detach().cpu().clone() for k,v in baseline.state_dict().items()}
        setup();model=Model(task,arm).cuda()
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v.cpu(),reference[k]) for k,v in model.state_dict().items())
        archived=load(BASE/'results'/(task+'_full')/'initial.pt')
        assert all(torch.equal(v,archived[k]) for k,v in reference.items())
        torch.cuda.reset_peak_memory_stats()
        if arm in ('full','thumbnail','uniform_budget'):
            with torch.no_grad():assert torch.equal(model(g),baseline(g)),arm
        del baseline,reference,archived
        output=model(g);loss=output.square().sum() if task=='local' else (output-.73).square()
        loss.backward();assert torch.isfinite(output).all()
        expected={'position':model.use_position,'path_position':model.use_position,
                  'scale':model.use_scale,'joint':True,'graph':True,'point':True}
        for name,enabled in expected.items():
            grad=getattr(model,name)[0].weight.grad
            if enabled:assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,(arm,name)
            else:assert grad is None,(arm,name)
        disabled=[name for name,on in expected.items() if not on]
        with torch.no_grad():
            before=model(g)
            for name in disabled:
                for param in getattr(model,name).parameters():param.add_(17.)
            assert torch.equal(before,model(g)),(arm,'disabled module leaked')
        assert model.processor.use_position==model.use_position
        checks[task+'_'+arm]=dict(initialization_and_rng_equal=True,gradient_gates_passed=True,
            disabled_parameter_invariance=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del g,model,output,loss;torch.cuda.empty_cache()
    audit=json.loads((ROOT/'smoke_cache/sampling_audit.json').read_text())
    for row in audit:
        assert row['skeleton_unchanged'] and row['origin_unchanged']
        if row['arm']=='uniform_budget':assert row['draws']==row['reference_draws']
        else:assert row['draws']==min(4096,row['source_points'])
    # Same matched-budget sampler as V23 thumbnail prefix for the 4096 arm.
    from .prepare import sampled_geometry
    from tools.cpsf_v21_segment_units.io import SOURCE
    import numpy as np
    r=next(r for r in records if r['task']=='global' and r['modal']=='joint')
    ref=load(r['path'])
    with np.load(SOURCE/'inputs'/(r['cell']+'.npz')) as x:
        _,a=sampled_geometry(x,'thumbnail',ref)
        ids=np.sort(np.random.default_rng(42).choice(len(x['surface_xyz_um']),min(4096,len(x['surface_xyz_um'])),replace=False))
        import hashlib
        assert a['sampling_indices_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,sampling_audit=audit,
        unchanged_full_bitwise_equal=True,complete_skeleton_preserved=True,
        training_samples_only=True,accuracy_claim=False))


if __name__=='__main__':main()

import json
import torch
from tools.cpsf_v21_segment_units.io import setup, load, save
from tools.cpsf_v21_segment_units.model import add
from tools.cpsf_v23_scale_ablation.model import Model as Original
from .data import adapt
from .model import Model, ARMS
from .configure import ROOT, V23
from .train import move


def main():
    setup(); records=json.loads((ROOT/'smoke_cache/manifest.json').read_text())['records']; checks={}
    for task in ('local','global'):
        r=next(r for r in records if r['task']==task and r['modal']=='joint' and r['split']=='train')
        raw=load(r['path']);g=adapt(raw)
        for k in raw:
            if k in ('coarse_owner','ncoarse'):continue
            if isinstance(raw[k],torch.Tensor):assert torch.equal(raw[k],g[k]),k
            else:assert raw[k]==g[k],k
        assert g['ncoarse']>=min(4,g['nregions'])
        g=move(g); reference=load(V23/'results'/(task+'_full')/'initial.pt')
        for arm in ARMS:
            setup();model=Model(task,arm).cuda()
            assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in reference.items())
            with torch.no_grad():
                if model.shared_position:
                    other=Original(task,'full').cuda();other.load_state_dict(reference)
                else:
                    other=Model(task,'no_shared_position').cuda();other.load_state_dict(model.state_dict())
                torch.testing.assert_close(model(g),other(g),rtol=0,atol=0);del other
                original_wp=g['pw']*g['patch_mass'][g['pj']]
                torch.testing.assert_close(model.incidence_weights(g),original_wp,rtol=0,atol=0)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            y=model(g);loss=y.square().sum() if task=='local' else (y-.73).square()
            loss.backward();assert torch.isfinite(y).all()
            for name,on in [('position',model.shared_position),('scale',True),('path_position',True),('relation',True)]:
                grad=getattr(model,name)[0].weight.grad
                if on:assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,(task,arm,name)
                else:assert grad is None,(task,arm,name)
            grad=model.routing_weight.grad
            if model.use_routing:assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,(task,arm,'routing')
            else:assert grad is None
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                if model.use_routing:
                    model.routing_weight.copy_(g['pos'].new_tensor([.7,-.4,.2,.3]))
                    weights=model.incidence_weights(g);alpha=weights/g['surface_mass'][g['pr']].clamp_min(1e-12)
                    assert (weights>=0).all() and torch.isfinite(weights).all()
                    torch.testing.assert_close(add(alpha,g['pr'],g['nregions'])[g['has_p']],torch.ones_like(g['surface_mass'][g['has_p']]),rtol=2e-5,atol=2e-5)
                    stats=model.routing_stats(g);assert stats['region_mean_tv']>1e-5
                    scaled=dict(g,patch_mass=g['patch_mass']*3.,surface_mass=g['surface_mass']*3.)
                    torch.testing.assert_close(model.incidence_weights(scaled)/3.,weights,rtol=2e-5,atol=2e-5)
                    shifted=dict(g);shift=g['pos'].new_tensor([.25,-.5,.125])
                    for k in ('pos','s','query'):
                        if k in g:shifted[k]=g[k]+shift
                    torch.testing.assert_close(model(g),model(shifted),rtol=1e-4,atol=1e-5)
            checks[task+'_'+arm]=dict(passed=True,initial_baseline_equivalence=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del model,opt,y,loss;torch.cuda.empty_cache()
        del g;torch.cuda.empty_cache()
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,fine_support_unchanged=True,
        baseline_equals_v23_repaired_hierarchy=True,zero_routing_equals_control=True,
        positive_normalized_attention=True,measure_rescaling_invariance=True,
        spatial_routing_changes_attention=True,translation_invariance=True,
        disabled_branch_gradient_isolation=True,accuracy_claim=False))


if __name__=='__main__':main()

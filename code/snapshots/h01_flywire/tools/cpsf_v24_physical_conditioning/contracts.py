import json
import torch
from tools.cpsf_v21_segment_units.io import setup,load,save
from tools.cpsf_v23_scale_ablation.model import Model as Original
from .data import adapt
from .model import Model,ARMS
from .configure import ROOT,V23
from .train import move

def main():
    setup();records=json.loads((ROOT/'smoke_cache/manifest.json').read_text())['records'];checks={}
    for task in ('local','global'):
        r=next(r for r in records if r['task']==task and r['modal']=='joint')
        raw=load(r['path']);g=adapt(raw)
        for k in raw:
            if k in ('coarse_owner','ncoarse'):continue
            if isinstance(raw[k],torch.Tensor):assert torch.equal(raw[k],g[k]),k
            else:assert raw[k]==g[k],k
        assert g['ncoarse']>=min(4,g['nregions'])
        assert len(torch.unique(g['coarse_owner']))==g['ncoarse']
        g=move(g);reference=load(V23/'results'/(task+'_full')/'initial.pt')
        for arm in ARMS:
            setup();model=Model(task,arm).cuda()
            assert all(torch.equal(v.cpu(),reference[k]) for k,v in model.state_dict().items() if k in reference)
            with torch.no_grad():
                old=Original(task,'full' if arm=='hierarchy_only' else 'no_position_scale').cuda();old.load_state_dict(reference)
                torch.testing.assert_close(model(g),old(g),rtol=0,atol=0)
                del old
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                opt.zero_grad(set_to_none=True);y=model(g)
                loss=y.square().sum() if task=='local' else (y-.73).square()
                loss.backward();assert torch.isfinite(y).all()
                if step==1:
                    for name,on in [('position',model.use_position),('scale',model.use_scale)]:
                        grad=getattr(model,name)[0].weight.grad
                        if on:assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,(task,arm,name)
                        else:assert grad is None
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            if arm=='full':
                with torch.no_grad():
                    model.position_alpha.fill_(.4);model.scale_alpha.fill_(-.2)
                    shift=g['pos'].new_tensor([.25,-.5,.125]);translated=dict(g)
                    for k in ('pos','s','query'):
                        if k in g:translated[k]=g[k]+shift
                    torch.testing.assert_close(model(g),model(translated),rtol=1e-4,atol=1e-5)
            checks[task+'_'+arm]=dict(passed=True,initial_baseline_equivalence=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del model,opt,y,loss;torch.cuda.empty_cache()
        del g;torch.cuda.empty_cache()
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,fine_support_unchanged=True,
        hierarchy_nonempty=True,initial_full_equals_neither=True,hierarchy_only_equals_v23=True,
        translation_equivariance=True,accuracy_claim=False))

if __name__=='__main__':main()

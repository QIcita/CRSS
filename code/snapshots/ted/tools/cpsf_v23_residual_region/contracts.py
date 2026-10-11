"""Real training geometry checks without consulting development outcomes."""
import json
import torch
from tools.cpsf_v21_segment_units.io import setup, load, save, sha
from tools.cpsf_v22_shared_region.model import Model as OldModel
from .configure import ROOT, BASE
from .model import Model
from .train import move


def main():
    setup()
    source_contracts = json.loads((BASE/'contracts.json').read_text())
    assert source_contracts['passed']
    records = json.loads((ROOT/'smoke_cache/manifest.json').read_text())['records']
    assert all(r['split'] == 'train' for r in records)
    checks = {}
    for task in ('local', 'global'):
        for arm in ('full', 'mean_only', 'no_fusion', 'point', 'skeleton'):
            modal = arm if arm in ('point','skeleton') else 'joint'
            row = next(r for r in records if r['task']==task and r['modal']==modal)
            assert sha(row['path']) == row['sha256']
            g = move(load(row['path']))
            setup(); model = Model(task, arm).cuda()
            reference = load(BASE/'results'/(task+'_full')/'initial.pt')
            assert all(torch.equal(v.cpu(), reference[k]) for k,v in model.state_dict().items())
            if arm in ('no_fusion','point','skeleton'):
                old = OldModel(task, arm).cuda(); old.load_state_dict(model.state_dict())
                with torch.no_grad(): assert torch.equal(model(g),old(g)), arm
                del old
            y = model(g)
            loss = y.square().sum() if task=='local' else (y-.73).square()
            loss.backward()
            assert torch.isfinite(y).all()
            for name in ('position','scale'):
                grad = getattr(model,name)[0].weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0, name
            if arm=='full':
                for name in ('graph','point','joint'):
                    grad=getattr(model,name)[0].weight.grad
                    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
            else:
                assert model.joint[0].weight.grad is None
            checks[task+'_'+arm]=dict(finite_backward=True,initialization_bitwise_equal=True,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),cell=row['cell'])
            del model,g,y,loss,reference
            torch.cuda.empty_cache()
    # A zero residual must be exactly mean_only; unmatched evidence must be identity.
    model=Model('local','full').cuda(); mean=Model('local','mean_only').cuda()
    a=torch.randn(4,96,device='cuda'); b=torch.randn_like(a)
    hg=torch.tensor([1,1,0,0],dtype=torch.bool,device='cuda')
    hp=torch.tensor([1,0,1,0],dtype=torch.bool,device='cuda')
    a=a*hg[:,None]; b=b*hp[:,None]
    combined=model.combine(a,b,hg,hp)
    assert torch.equal(combined[1:],(a+b)[1:])
    # Missing-modality values cannot leak because stems mask them before combine.
    with torch.no_grad():
        for p in model.joint.parameters(): p.zero_()
    assert torch.equal(model.combine(a,b,hg,hp),mean.combine(a,b,hg,hp))
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,
        inherited_geometry_contract_sha256=sha(BASE/'contracts.json'),
        identity_unmatched=True,zero_residual_equals_mean=True,
        scale_position_gradients=True,unchanged_controls_bitwise_equal=True,
        training_samples_only=True,accuracy_claim=False))


if __name__=='__main__':
    main()

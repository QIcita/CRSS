import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup,sha,save
from tools.cpsf_flywire_v23.model import Model
from tools.cpsf_flywire_v23.data import load
from .configure import ROOT,ARMS
from tools.cpsf_flywire_balanced.loss import balanced_loss,weights_from_counts
from .train import move


def main():
    setup();p=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest
    counts=torch.tensor(p['loss']['train_counts']);w=weights_from_counts(counts)
    torch.testing.assert_close((w*counts).sum()/counts.sum(),torch.tensor(1.),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(w*counts,torch.full((6,),counts.sum().item()/6),rtol=1e-6,atol=1e-5)
    # Actual single-cell accumulation must equal a batch mean, not cancel w_y.
    x=torch.linspace(-1,1,36,dtype=torch.float64).reshape(6,6).requires_grad_();y=torch.arange(6)
    batch=balanced_loss(x,y,w.double());gb=torch.autograd.grad(batch,x)[0]
    separate=x.detach().clone().requires_grad_()
    loss=sum(balanced_loss(separate[i:i+1],y[i:i+1],w.double())/6 for i in range(6))
    gs=torch.autograd.grad(loss,separate)[0];torch.testing.assert_close(gb,gs,rtol=1e-12,atol=1e-12)
    for c in range(6):
        z=x[c:c+1].detach().clone().requires_grad_()
        raw=F.cross_entropy(z,y[c:c+1],label_smoothing=.05)
        raw_grad=torch.autograd.grad(raw,z,retain_graph=True)[0]
        weighted=balanced_loss(z,y[c:c+1],w.double())
        grad=torch.autograd.grad(weighted,z)[0]
        torch.testing.assert_close(grad,raw_grad*w[c].double(),rtol=1e-12,atol=1e-12)
    z=x.detach().clone().requires_grad_()
    ones_loss=balanced_loss(z,y,torch.ones(6,dtype=torch.float64))
    old=F.cross_entropy(z,y,label_smoothing=.05)
    torch.testing.assert_close(ones_loss,old,rtol=1e-12,atol=1e-12)
    records=json.loads((ROOT/'smoke_records.json').read_text());assert all(r['split']=='train' for r in records)
    assert set(r['coarse_target'] for r in records)==set(range(6))
    from .budget import select_round
    import numpy as np
    manifest=json.loads((ROOT/'cache/manifest.json').read_text());train=[r for r in manifest['records'] if r['split']=='train'];seen=set();rng=np.random.RandomState(42)
    for _ in range(50):seen.update(train[i]['root_id'] for i in select_round(train,rng))
    assert len(seen)==len(train) and 50*((4975+3)//4)==62200
    largest=max((r for r in records if not r['legacy']),key=lambda r:r.get('relation_pairs',0))
    checks={}
    for arm in ARMS:
        setup();model=Model(6,arm).cuda();ref=p['reference_runs'][arm]
        for fn,digest in ref['sha256'].items():assert sha(Path(ref['path'])/fn)==digest
        old=torch.load(Path(ref['path'])/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v.cpu(),old[k]) for k,v in model.state_dict().items())
        r=largest;assert sha(r['path'])==r['sha256'];g=move(load(r['path']))
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
        pred=model(g);target=torch.tensor([r['coarse_target']],device='cuda')
        loss=balanced_loss(pred[None],target,w.cuda());loss.backward()
        for name in ('position','scale'):
            grad=getattr(model,name)[0].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        assert (model.joint[0].weight.grad is not None)==(arm=='full')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
        checks[arm]=dict(passed=True,initial_matches_original=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del model,opt,g,pred,loss;torch.cuda.empty_cache()
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,train_only_weights=True,
        equal_class_total_weight=True,mean_training_weight_one=True,
        sample_accumulation_equals_batch_gradient=True,weights_do_not_cancel=True,
        unit_weights_match_original_loss=True,all_classes_in_training_smoke=True,
        test_used=False,accuracy_claim=False))


if __name__=='__main__':main()

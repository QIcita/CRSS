import json
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup
from tools.cpsf_flywire_v23.model import Model
from tools.cpsf_flywire_v23.data import load
from tools.cpsf_flywire_balanced.loss import balanced_loss
from .common import ROOT,COHORT,ARMS,save,sha,verify_sources
from .train import move

def main():
    p=json.loads((ROOT/'protocol.json').read_text());verify_sources(ROOT,p)
    parent=json.loads((COHORT/'protocol.json').read_text());verify_sources(COHORT,parent)
    assert sha(COHORT/'protocol.json')==p['inherited_protocol_sha256']
    x=torch.linspace(-1,1,36,dtype=torch.float64).reshape(6,6).requires_grad_();y=torch.arange(6)
    ordinary=F.cross_entropy(x,y,label_smoothing=.05)
    weighted=balanced_loss(x,y,torch.ones(6,dtype=torch.float64))
    torch.testing.assert_close(ordinary,weighted,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(ordinary,x,retain_graph=True)[0],
        torch.autograd.grad(weighted,x)[0],rtol=1e-12,atol=1e-12)
    samples=json.loads((ROOT/'smoke_records.json').read_text());assert all(r['split']=='train' for r in samples)
    largest=max(samples,key=lambda r:r.get('relation_pairs',0));checks={}
    for arm in ARMS:
        setup();model=Model(6,arm).cuda();ref=p['reference_runs'][arm]
        for name,digest in ref['sha256'].items():assert sha(Path(ref['path'])/name)==digest
        initial=torch.load(Path(ref['path'])/'initial.pt',map_location='cpu',weights_only=False)
        assert model.state_dict().keys()==initial.keys()
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        assert sha(largest['path'])==largest['sha256'];g=move(load(largest['path']))
        torch.cuda.reset_peak_memory_stats();z=model(g);loss=F.cross_entropy(z[None],torch.tensor([largest['coarse_target']],device='cuda'),label_smoothing=.05)
        loss.backward()
        for name in ('scale','position','graph','point'):
            grad=getattr(model,name)[0].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        assert (model.joint[0].weight.grad is not None)==(arm=='full')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        checks[arm]=dict(initial_exact=True,physical_codes_gradient=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del model,g,z,loss;torch.cuda.empty_cache()
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,unit_weight_CE_and_gradient_match=True,
        inherited_source_exact=True,train_only=True,test_used=False,accuracy_claim=False))

from pathlib import Path
if __name__=='__main__':main()

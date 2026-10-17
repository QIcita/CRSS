"""Verify blocked/rematerialized reduction against the original operator."""
import json
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha
from .data import ROOT,load
from .model import Model
from .train import move

def main():
    rows=json.loads((ROOT/'profile_cache/manifest.json').read_text())['records']
    row=min((r for r in rows if r['variant']=='joint'),key=lambda r:r['bytes'])
    g=move(load(row['path']));outputs=[];gradients=[]
    for chunk in (10**12,128):
        setup();model=Model(6,'full').cuda();model.chunk_size=chunk
        z=model(g);loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([row['coarse_target']],device='cuda'),label_smoothing=.05)
        loss.backward();outputs.append(z.detach().cpu());gradients.append({n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters()})
        del model,z,loss;torch.cuda.empty_cache()
    torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-5,atol=1e-6)
    errors={}
    for name,a in gradients[0].items():
        b=gradients[1][name]
        if a is None:assert b is None;continue
        torch.testing.assert_close(a,b,rtol=2e-4,atol=1e-6)
        errors[name]=float((a-b).abs().max())
    save(ROOT/'chunk_gradient_equivalence.json',dict(passed=True,training_root=row['root_id'],
        output_max_abs=float((outputs[0]-outputs[1]).abs().max()),gradient_max_abs=max(errors.values()),
        source_sha256=sha(ROOT/'tools/cpsf_flywire_v23/model.py'),original_chunk_size=10**12,test_chunk_size=128,
        meaning='same operator within floating point reduction tolerance; not bitwise identity'))

if __name__=='__main__':main()

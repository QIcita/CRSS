import argparse
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup
from .common import *
from .model import Model
from .train import move,smoke_rows,expected_initial

def run(task,arm):
    setup();verify();rr,_=rows(task,arm);sample=smoke_rows([r for r in rr if r['split']=='train'],task)
    model=Model(task,arm).cuda();ref=expected_initial(task)
    assert not any('gate' in n for n,_ in model.named_parameters())
    assert all(torch.equal(v.cpu(),ref[k]) for k,v in model.state_dict().items())
    for r in sample:
        assert sha(r['path'])==r['sha256'];g=move(load_graph(r,task))
        if arm=='shuffle_regions':
            ids=torch.where(g['has_g']&g['has_p'])[0];perm=g['surface_content_permutation']
            assert torch.equal(perm.sort().values,torch.arange(g['nregions'],device=perm.device))
            if len(ids)>1:assert (perm[ids]!=ids).all()
        with torch.no_grad():assert torch.isfinite(model(g)).all()
    r=max(sample,key=lambda r:Path(r['path']).stat().st_size);g=move(load_graph(r,task));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);z=model(g)
    loss=(z-r['target']).square() if task=='h01_global' else F.cross_entropy(z[None],torch.tensor([r['target']],device='cuda'),label_smoothing=.05)
    loss.backward();active={n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any())}
    assert any(n.startswith('position.') for n in active)==model.use_position
    assert any(n.startswith('scale.') for n in active)==model.use_scale
    assert any(n.startswith('processor.context.') for n in active)==(arm!='no_context')
    assert any(n.startswith('joint.') for n in active)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    save(ROOT/'contracts'/(task+'_'+arm+'.json'),dict(passed=True,train_only=True,test_used=False,
        gate_parameters_absent=True,active_parameters=sorted(active),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--arm',required=True);a=ap.parse_args();run(a.task,a.arm)

import argparse,json
import torch
from torch.nn import functional as F
from tools.cpsf_v21_segment_units.io import setup
from .common import *
from .model import Model
from .train import move,smoke_rows,expected_initial

def run(task,arm):
    setup();p=verify();rr,ready=rows(task,arm);sample=smoke_rows([r for r in rr if r['split']=='train'],task)
    model=Model(task,arm).cuda();ref=expected_initial(task)
    assert all(torch.equal(v.cpu(),ref[k]) for k,v in model.state_dict().items() if k in ref)
    checks=[]
    for r in sample:
        assert sha(r['path'])==r['sha256'];g=move(load_graph(r,task))
        if arm=='point':assert not g['has_g'].any() and len(g['s'])==0
        if arm=='skeleton':assert not g['has_p'].any() and len(g['pf'])==0
        if task.startswith('h01'):assert g['ncoarse']>=min(4,g['nregions'])
        with torch.no_grad():assert torch.isfinite(model(g)).all()
        checks.append(r['key'])
    # Use the largest selected real training graph for backward admission.
    r=max(sample,key=lambda r:Path(r['path']).stat().st_size);g=move(load_graph(r,task))
    if arm in ('full','gate_global'):
        control=Model(task,'mean_only').cuda();control.load_state_dict(model.state_dict())
        with torch.no_grad():torch.testing.assert_close(model(g),control(g),rtol=0,atol=0)
        del control
    if arm=='fixed_residual':
        if task=='flywire_coarse':
            from tools.cpsf_flywire_v23.model import Model as Original
            control=Original(6,'full').cuda()
        else:
            from tools.cpsf_v25_position_routing.model import Model as Original
            control=Original('local' if task=='h01_local' else 'global','baseline').cuda()
        own=control.state_dict();own.update({k:v for k,v in model.state_dict().items() if k in own});control.load_state_dict(own)
        with torch.no_grad():torch.testing.assert_close(model(g),control(g),rtol=0,atol=0)
        del control
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);active=set();torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        opt.zero_grad(set_to_none=True);z=model(g)
        loss=(z-r['target']).square() if task=='h01_global' else F.cross_entropy(z[None],torch.tensor([r['target']],device='cuda'),label_smoothing=.05)
        loss.backward();active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    assert any(n.startswith('position.') for n in active)==model.use_position
    assert any(n.startswith('scale.') for n in active)==model.use_scale
    assert any(n.startswith('graph.') for n in active)==(arm!='point')
    assert any(n.startswith('point.') for n in active)==(arm!='skeleton')
    gated=arm not in ('fixed_residual','mean_only','no_fusion','point','skeleton')
    assert ('gate_alpha' in active)==gated
    assert any(n.startswith('region_gate.') for n in active)==(gated and arm!='gate_global')
    save(ROOT/'contracts'/(task+'_'+arm+'.json'),dict(passed=True,train_only=True,sample_keys=checks,
        active_parameters=sorted(active),peak_gpu_bytes=torch.cuda.max_memory_allocated(),test_used=False,protocol_sha256=sha(ROOT/'protocol.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--arm',required=True);a=ap.parse_args();run(a.task,a.arm)

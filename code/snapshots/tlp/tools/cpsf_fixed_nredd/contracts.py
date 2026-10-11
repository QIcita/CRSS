import argparse,time
import numpy as np
from .common import *
from .model import Model
from .train import state_digest,average_precision,auc
from tools.cpsf_fixed_core.model import H01Model

def main(task):
    setup();verify();results={};initial=None
    assert abs(average_precision([1,0,1],[.8,.8,.1])-7/12)<1e-12
    assert abs(auc([1,0],[.5,.5])-.5)<1e-12
    for arm in ARMS:
        if task=='ted' and arm in ('thumbnail','uniform_budget'):continue
        setup();m=Model(task,arm).cuda();h=state_digest(m)
        if initial is None:initial=h
        assert h==initial,('unmatched initialization',arm)
        rr=json.loads((ROOT/'pilot'/task/modal(arm)/'manifest.json').read_text())['records']
        item=load(rr[0]['path']);g=move(item['g']);torch.cuda.reset_peak_memory_stats();start=time.time()
        z=m(g);assert torch.isfinite(z).all()
        if task=='tlp':assert torch.allclose(z,z.T,atol=1e-6,rtol=1e-6)
        z.square().mean().backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        grads={n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any())}
        assert any(n.startswith('graph.') for n in grads)==(arm!='point')
        assert any(n.startswith('point.') for n in grads)==(arm!='skeleton')
        assert any(n.startswith('position.') for n in grads)==m.use_position
        assert any(n.startswith('scale.') for n in grads)==m.use_scale
        if arm=='no_context':assert not any(n.startswith(('processor.qkv.','processor.context.')) for n in grads)
        perm=g['surface_content_permutation'];ids=torch.where(g['has_g']&g['has_p'])[0]
        assert torch.equal(torch.sort(perm).values,torch.arange(g['nregions'],device='cuda'))
        if len(ids)>1:assert (perm[ids]!=ids).all()
        if arm=='full' and task=='ted':
            # Actual clean-core readout and new TED query pooling agree.
            reference=H01Model('h01_local','full').cuda();state=m.state_dict();reference.load_state_dict({k:(state[k] if not k.startswith('head.') else v) for k,v in reference.state_dict().items()})
            gg=dict(g,query=g['query_xyz'][0]);reference.eval();m.eval()
            with torch.no_grad():
                actual=reference.embedding(gg);x,w,a=m.streams(g)[0];current=m.readout(m.query_pool(x,g,w,a))[0]
                assert torch.allclose(actual,current,atol=2e-5,rtol=2e-5)
            del reference
        results[arm]=dict(finite=True,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),regions=g['nregions'],nodes=g['raw_nodes'],parameters=sum(v.numel() for v in m.parameters()),initial_sha256=h)
        del m,g,z;torch.cuda.empty_cache();print(task,arm,results[arm],flush=True)
    save(ROOT/'contracts'/(task+'.json'),dict(passed=True,task=task,results=results,train_only=True,protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);a=p.parse_args();main(a.task)

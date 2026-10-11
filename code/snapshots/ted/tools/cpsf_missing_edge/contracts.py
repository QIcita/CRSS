"""Validate task universe, exclusion gradients, ordering and actual models."""
import numpy as np
from .common import *
from .objective import missing_edge_loss
from .train import state_digest
from tools.cpsf_fixed_nredd.model import Model

def main():
    setup();p=verify()
    # Exactly four eligible unordered pairs; known labels and excluded gradients.
    target=torch.tensor([[1.,1.,1.,0.],[1.,1.,0.,1.],[1.,0.,1.,0.],[0.,1.,0.,1.]])
    observed=torch.eye(4,dtype=torch.bool);observed[0,1]=observed[1,0]=True;observed[2,3]=observed[3,2]=True
    z=torch.zeros((4,4),requires_grad=True)
    loss=missing_edge_loss(z,target,observed)
    # Eligible (0,2),(0,3),(1,2),(1,3): two positives, two negatives.
    assert abs(float(loss)-np.log(2))<1e-6
    loss.backward();mask=torch.triu(~observed,1);assert (z.grad[~mask]==0).all() and (z.grad[mask]!=0).all()
    changed=target.clone();changed[~mask]=1-changed[~mask]
    assert torch.equal(loss.detach(),missing_edge_loss(z.detach(),changed,observed))
    for labels in [torch.zeros((4,4)),torch.ones((4,4))]:assert torch.isfinite(missing_edge_loss(z.detach(),labels,observed))
    torch.manual_seed(42);z=torch.randn(4,4);z=(z+z.T)/2
    ix=torch.tensor([2,0,3,1]);assert torch.allclose(missing_edge_loss(z,target,observed),missing_edge_loss(z[ix][:,ix],target[ix][:,ix],observed[ix][:,ix]))
    # Real train-only model tests and exact frozen parent initialization.
    results={};fingerprint=None
    for arm in ARMS:
        setup();m=Model('tlp',arm).cuda();h=state_digest(m)
        parent=json.loads((PARENT/'results'/('tlp_'+arm)/'identity.json').read_text())
        assert h==parent['initial_state_sha256']
        if fingerprint is None:fingerprint=h
        assert h==fingerprint
        r=json.loads((ROOT/'pilot/tlp'/modal(arm)/'manifest.json').read_text())['records'][0]
        item=load(r['path']);g=move(item['g']);yy=item['target'].cuda();obs=item['observed'].cuda()
        torch.cuda.reset_peak_memory_stats();out=m(g);loss=missing_edge_loss(out,yy,obs)
        assert torch.isfinite(loss) and torch.allclose(out,out.T,atol=1e-6)
        loss.backward();grads={n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any())}
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        assert any(n.startswith('graph.') for n in grads)==(arm!='point')
        assert any(n.startswith('point.') for n in grads)==(arm!='skeleton')
        assert any(n.startswith('scale.') for n in grads)==(arm!='no_scale')
        assert any(n.startswith('position.') for n in grads)
        assert any(n.startswith('joint.') for n in grads)==(arm in ('full','no_scale'))
        results[arm]=dict(initial_state_sha256=h,finite=True,loss=float(loss),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del m,g,out,loss;torch.cuda.empty_cache()
    # Audit every real training target and cache hash; train/val grouping unchanged.
    audit=[]
    for mode in ('joint','skeleton','point'):
        rr,_=rows('tlp',mode if mode!='joint' else 'full')
        assert {r['group'] for r in rr if r['split']=='train'}.isdisjoint(r['group'] for r in rr if r['split']=='val')
        for r in rr:
            assert sha(r['path'])==r['sha256']
            if r['split']!='train':continue
            item=load(r['path']);y=item['target'];obs=item['observed'];mask=torch.triu(~obs,1)
            assert torch.equal(y,y.T) and torch.equal(obs,obs.T) and set(torch.unique(y).tolist())<={0.,1.}
            assert 'target' not in item['g'] and 'observed' not in item['g']
            audit.append(dict(mode=mode,key=r['key'],eligible=int(mask.sum()),positive=int(y[mask].sum())))
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),tests=['mask_reference','excluded_gradient_zero','excluded_labels_no_effect','degenerate_targets_finite','permutation_invariance','exact_parent_initialization','real_modality_gradients','train_val_group_disjoint','all_cache_hashes'],results=results,training_targets=audit,test_used=False))
    print('contracts passed',flush=True)
if __name__=='__main__':main()

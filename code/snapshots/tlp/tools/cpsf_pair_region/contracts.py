import time
import numpy as np
from .common import *
from .model import Model
from .data import support
from .train import state_digest
from tools.cpsf_fixed_nredd.model import Model as Baseline
from tools.cpsf_missing_edge.objective import missing_edge_loss

def main():
    setup();verify();results={}
    # A midpoint outside both endpoint balls must receive support and gradient.
    tiny=dict(query_xyz=torch.tensor([[-6.,0,0],[6.,0,0]]),pos=torch.tensor([[-6.,0,0],[0.,0,0],[6.,0,0]]),radius=2.)
    sg=support(tiny,'synthetic');assert 1 in sg['bridge_col'].tolist()
    assert set(sg['bridge_col'].tolist())=={1}
    m=Model('tlp','full').cuda();x=torch.randn(3,96,device='cuda',requires_grad=True)
    pooled,valid=m.corridor_pool(x,move(sg),torch.ones(3,device='cuda'),torch.ones(3,device='cuda',dtype=torch.bool))
    pooled.sum().backward();assert valid.all() and x.grad[1].abs().sum()>0 and x.grad[[0,2]].abs().sum()==0
    reverse=dict(tiny,query_xyz=tiny['query_xyz'].flip(0));rev=support(reverse,'synthetic')
    for key in ['bridge_row','bridge_col','bridge_weight']:assert torch.equal(sg[key],rev[key])
    empty=support(dict(tiny,pos=torch.tensor([[0.,100.,0.]])),'empty');assert not len(empty['bridge_row'])
    del m,x,pooled
    fingerprint=None
    for arm in ARMS:
        setup();m=Model('tlp',arm).cuda();h=state_digest(m)
        old=json.loads((PARENT/'results'/('tlp_'+parent_arm(arm))/'identity.json').read_text());assert h==old['initial_state_sha256']
        if fingerprint is None:fingerprint=h
        assert h==fingerprint
        rr=json.loads((ROOT/'pilot/tlp'/modal(arm)/'manifest.json').read_text())['records']
        measurements=[]
        for r in (rr[0],rr[-1]):
            item=load(r['path']);g=move(item['g']);torch.cuda.reset_peak_memory_stats();t=time.time()
            m.zero_grad(set_to_none=True);z=m(g);assert torch.allclose(z,z.T,atol=2e-5)
            loss=missing_edge_loss(z,item['target'].cuda(),item['observed'].cuda());assert torch.isfinite(loss)
            loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            measurements.append(dict(nodes=len(z),regions=g['nregions'],incidences=len(g['bridge_row']),seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del z,loss
            if r==rr[0]:
                # Exactly the same parameters; disabling the new path recovers stage1.
                base=Baseline('tlp',parent_arm(arm)).cuda();base.load_state_dict(m.state_dict());m.use_corridor=False
                with torch.no_grad():
                    a=m(g);b=base(g);assert torch.equal(a,b)
                    m.use_corridor=True
                    gg=dict(g,bridge_row=g['bridge_row'][:0],bridge_col=g['bridge_col'][:0],bridge_weight=g['bridge_weight'][:0])
                    assert torch.allclose(m(gg),b,atol=3e-5,rtol=3e-5)
                    if arm=='full':
                        # Rebuild support under a node permutation without touching encoder input.
                        perm=torch.arange(len(a)-1,-1,-1,device='cuda');cpu=dict(item['g']);cpu['query_xyz']=cpu['query_xyz'].flip(0)
                        cpu.update(support(cpu,r['key']));reversed_z=m(move(cpu));original=m(g)
                        assert torch.allclose(reversed_z,original[perm][:,perm],atol=3e-5,rtol=3e-5)
                del base,a,b
            del g;torch.cuda.empty_cache()
        results[arm]=dict(initial_state_sha256=h,parameters=sum(v.numel() for v in m.parameters()),measurements=measurements)
        print(arm,results[arm],flush=True);del m;torch.cuda.empty_cache()
    # Only geometry fields were added; targets, observations and every original tensor are identical.
    audits={}
    for mode in ['joint','skeleton','point']:
        rr,_=rows('tlp',mode if mode!='joint' else 'full')
        for r in rr:
            assert sha(r['path'])==r['sha256'] and sha(r['parent_cache_path'])==r['parent_cache_sha256']
            item=load(r['path']);old=load(r['parent_cache_path']);g=item['g']
            for key,value in old['g'].items():
                assert torch.equal(value,g[key]) if isinstance(value,torch.Tensor) else value==g[key]
            for key in ['target','observed','pairs']:assert torch.equal(item[key],old[key])
            assert 'target' not in g and 'observed' not in g
            assert g['pair_count']==len(g['query_xyz'])*(len(g['query_xyz'])-1)//2
            assert torch.equal(g['pair_lookup'],g['pair_lookup'].T)
        audits[mode]=len(rr)
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),results=results,
        tests=['midpoint_support_and_gradient','endpoint_reversal','all_parameters_match_stage1','disable_path_exact_stage1','empty_support_global_fallback','node_permutation_equivariance','finite_real_train_gradients','all_original_inputs_unchanged','all_pair_universe'],cache_audits=audits,test_used=False))
    print('contracts passed',flush=True)
if __name__=='__main__':main()

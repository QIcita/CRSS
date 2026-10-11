import numpy as np
from .common import *
from .descriptor import describe
from .model import Model
from .train import state_digest
from tools.cpsf_pair_region.model import Model as Baseline
from tools.cpsf_missing_edge.objective import missing_edge_loss

def main():
    setup();verify()
    # Well-separated synthetic support distinguishes shape even though singleton
    # voxel-relative descriptors would be identically zero.
    line=np.column_stack((np.linspace(-1,1,9),np.zeros((9,2))));a,info=describe(line);assert info['valid_fraction']==1 and np.allclose(a[:,0],1)
    plane=np.array([[i,j,0.] for i in [-1,0,1] for j in [-1,0,1]]);b,_=describe(plane);assert np.allclose(b[:,1],1)
    sphere=np.vstack((np.eye(3),-np.eye(3)));c,_=describe(sphere);assert np.allclose(c[:,2],1)
    isolated,info=describe([[0,0,0],[100,0,0]]);assert not isolated.any() and info['valid_fraction']==0
    duplicate,info=describe(np.repeat(line,3,axis=0));assert np.allclose(duplicate,np.repeat(a,3,axis=0)) and info['distinct_support']==len(line)
    rng=np.random.default_rng(42);points=rng.normal(size=(11,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
    x,_=describe(points);y,_=describe(points@q+[2,3,4]);assert np.allclose(x,y,atol=1e-6)
    ix=rng.permutation(len(points));y,_=describe(points[ix]);assert np.allclose(y,x[ix],atol=1e-6)
    results={};fingerprint=None
    for arm in ARMS:
        setup();m=Model('tlp',arm).cuda();h=state_digest(m);base_dir=PARENT/'results'/('tlp_'+parent_arm(arm))
        assert h==json.loads((base_dir/'identity.json').read_text())['initial_state_sha256']
        if fingerprint is None:fingerprint=h
        assert h==fingerprint
        rr=json.loads((ROOT/'pilot/tlp'/modal(arm)/'manifest.json').read_text())['records'];measurements=[]
        for record in [rr[0],rr[-1]]:
            item=load(record['path']);g=move(item['g']);torch.cuda.reset_peak_memory_stats();m.zero_grad(set_to_none=True)
            z=m(g);assert torch.isfinite(z).all() and torch.allclose(z,z.T,atol=2e-5)
            loss=missing_edge_loss(z,item['target'].cuda(),item['observed'].cuda());loss.backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            grads={n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any())}
            assert any(n.startswith('point.') for n in grads)==(arm!='skeleton')
            assert any(n.startswith('graph.') for n in grads)==(arm!='point')
            assert any(n.startswith('scale.') for n in grads)==(arm!='no_scale')
            measurements.append(dict(nodes=len(z),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del z,loss
            if record==rr[0]:
                base=Baseline('tlp',parent_arm(arm)).cuda();base.load_state_dict(m.state_dict())
                with torch.no_grad():
                    m.use_descriptor=False;assert torch.equal(m(g),base(g));m.use_descriptor=True
                    if arm=='skeleton':assert torch.equal(m(g),base(g))
                    else:
                        pf=g['pf_neighborhood'].clone().requires_grad_(True);gg=dict(g,pf_neighborhood=pf)
                if arm!='skeleton':
                    out=m(gg);gradient=torch.autograd.grad(out.square().mean(),pf)[0];assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                    del out,gradient,pf,gg
                del base
            del g;torch.cuda.empty_cache()
        results[arm]=dict(initial_state_sha256=h,parameters=sum(v.numel() for v in m.parameters()),measurements=measurements)
        print(arm,results[arm],flush=True);del m;torch.cuda.empty_cache()
    audits={}
    for mode in ['joint','point','skeleton']:
        rr,_=rows('tlp',mode if mode!='joint' else 'full')
        for record in rr:
            assert sha(record['path'])==record['sha256'] and sha(record['parent_cache_path'])==record['parent_cache_sha256']
            item=load(record['path']);old=load(record['parent_cache_path']);g=item['g']
            for key,value in old['g'].items():assert torch.equal(value,g[key]) if isinstance(value,torch.Tensor) else value==g[key]
            for key in ['target','observed','pairs']:assert torch.equal(item[key],old[key])
            assert g['pf_neighborhood'].shape==g['pf'].shape and torch.isfinite(g['pf_neighborhood']).all()
            assert torch.equal(g['descriptor_permutation'].sort().values,torch.arange(len(g['pf'])))
        audits[mode]=len(rr)
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),results=results,cache_audits=audits,test_used=False,
        tests=['line_plane_scatter_geometry','isolated_support_stays_zero','near_coincident_deduplication','rigid_transform_well_separated','point_order_invariance','identical_parent_initialization','descriptor_disable_exact_parent','skeleton_exact_negative_control','new_descriptor_gradient','all_old_inputs_and_pair_readout_unchanged']))
    print('contracts passed',flush=True)
if __name__=='__main__':main()

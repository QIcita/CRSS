import time
import numpy as np
from .common import *
from .descriptor import describe
from .model import Model
from .train import state_digest
from tools.cpsf_deterministic.model import Model as ParentModel
from tools.cpsf_missing_edge.objective import missing_edge_loss

def geometry_contracts():
    rng=np.random.default_rng(42);points=rng.normal(size=(11,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
    a,valid,_=describe(points,radius=10)
    assert valid.all() and np.allclose(a[:,:3],(points-points.mean(0))/.5,atol=1e-6)
    assert np.allclose(a[:,3],np.linalg.norm(a[:,:3],axis=1),atol=1e-6)
    b,v,_=describe(points+[10,20,30],radius=10);assert np.array_equal(v,valid) and np.allclose(a,b,atol=1e-6)
    b,v,_=describe(points@q,radius=10);assert np.allclose(b[:,:3],a[:,:3]@q,atol=1e-6) and np.allclose(b[:,3],a[:,3],atol=1e-6)
    order=rng.permutation(len(points));b,_,_=describe(points[order],radius=10);assert np.allclose(b,a[order],atol=1e-6)
    b,_,s=describe(np.repeat(points,3,axis=0),radius=10);assert s['distinct_support']==len(points) and np.allclose(b,np.repeat(a,3,axis=0),atol=1e-6)
    z,v,_=describe([[0,0,0],[100,0,0]]);assert not v.any() and not z.any()
    z,v,_=describe(np.empty((0,3)));assert z.shape==(0,4) and len(v)==0

def main():
    setup();p=verify();geometry_contracts();begin=time.time();selected={};results={}
    for arm in ('full','point','skeleton'):
        rr,_=rows('tlp',arm);train=[r for r in rr if r['split']=='train']
        choices=[min(train,key=lambda r:r['query_nodes']),max(train,key=lambda r:r['query_nodes']),max(train,key=lambda r:len(load(r['path'])['g']['pf']))]
        selected[modal(arm)]=list({r['key']:r for r in choices}.values())
    for j in p['jobs']:
        arm,d=j['arm'],j['descriptor'];setup();m=Model('tlp',arm,d).cuda();h=state_digest(m)
        assert h==json.loads((parent_result(arm,d)/'identity.json').read_text())['initial_state_sha256']
        measurements=[]
        for r in selected[modal(arm)]:
            item=load(r['path']);g=move(item['g']);y=item['target'].cuda();obs=item['observed'].cuda();ref=None;torch.cuda.reset_peak_memory_stats()
            for repeat in range(4):
                m.zero_grad(set_to_none=True);z=m(g);loss=missing_edge_loss(z,y,obs);loss.backward()
                grads={n:v.grad.detach().cpu().clone() for n,v in m.named_parameters() if v.grad is not None}
                assert all(torch.isfinite(v).all() for v in grads.values())
                if ref is None:ref=(z.detach().cpu(),grads)
                else:assert torch.equal(z.detach().cpu(),ref[0]) and all(torch.equal(v,ref[1][n]) for n,v in grads.items())
                del z,loss
            assert any(n.startswith('scale.') and v.abs().sum()>0 for n,v in grads.items())==(arm!='no_scale')
            measurements.append(dict(key=r['key'],nodes=len(y),points=len(g['pf']),repeats=4,exact=True,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del g,y,obs,ref,grads
        results[job_key(arm,d)]=dict(parameters=sum(v.numel() for v in m.parameters()),initial_state_sha256=h,measurements=measurements)
        print(job_key(arm,d),measurements,flush=True);del m;torch.cuda.empty_cache()
    for arm in ARMS:
        setup();m=Model('tlp',arm,'voxel').cuda();base=ParentModel('tlp',arm,'voxel').cuda();base.load_state_dict(m.state_dict())
        for weights in ('initial','final'):
            if weights=='final':
                state=torch.load(parent_result(arm,'voxel')/'final.pt',map_location='cpu',weights_only=False)['model'];m.load_state_dict(state);base.load_state_dict(state)
            for r in selected[modal(arm)]:
                g=move(load(r['path'])['g'])
                with torch.no_grad():assert torch.equal(m(g),base(g))
                if arm=='skeleton':
                    for d in DESCRIPTORS:
                        m.directional_descriptor=d
                        with torch.no_grad():assert torch.equal(m(g),base(g))
                    m.directional_descriptor='voxel'
        del m,base;torch.cuda.empty_cache()
    audited=0
    for arm in ('full','point','skeleton'):
        rr,_=rows('tlp',arm)
        for r in rr:
            assert sha(r['path'])==r['sha256'] and sha(r['parent_cache_path'])==r['parent_cache_sha256']
            item=load(r['path']);old=load(r['parent_cache_path']);g=item['g']
            for k,v in old['g'].items():assert torch.equal(v,g[k]) if isinstance(v,torch.Tensor) else v==g[k]
            for k in ('target','observed','pairs'):assert torch.equal(item[k],old[k])
            assert g['pf_directional'].shape==g['pf'].shape and torch.isfinite(g['pf_directional']).all()
            valid=g['directional_valid'];assert torch.equal(g['pf_directional'][~valid],g['pf'][~valid])
            if valid.any():assert torch.allclose(g['pf_directional'][valid,3],g['pf_directional'][valid,:3].norm(dim=1),atol=1e-6)
            audited+=1
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),results=results,cache_audits=audited,seconds=time.time()-begin,test_used=False,
        tests=['real_centroid_offset_formula','translation_invariance','rotation_equivariance_xyz_and_invariant_norm','point_order_and_duplicate_support',
               'invalid_support_fallback','four_exact_repeats','initial_and_final_voxel_parent_exact','skeleton_invariant','old_inputs_unchanged','scale_encoding_retained']))
    print('contracts passed',flush=True)
if __name__=='__main__':main()

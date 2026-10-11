import time
from .common import *
from .model import Model,descriptor
from .train import state_digest
from tools.cpsf_deterministic.model import Model as ParentModel
from tools.cpsf_missing_edge.objective import missing_edge_loss

def main():
    setup();p=verify();begin=time.time();results={}
    old=torch.randn(8,4,device='cuda',requires_grad=True);new=torch.randn(8,4,device='cuda',requires_grad=True)
    assert descriptor(old,new,'voxel') is old and descriptor(old,new,'neighborhood') is new
    for d in ('shape_only','extent_only'):
        v=descriptor(old,new,d);a,b=torch.autograd.grad(v.sum(),(old,new))
        first_old=d=='extent_only'
        assert torch.equal(v[:,:3],old[:,:3] if first_old else new[:,:3])
        assert torch.equal(v[:,3:],new[:,3:] if first_old else old[:,3:])
        mask=torch.tensor([first_old,first_old,first_old,not first_old],device='cuda',dtype=old.dtype).expand_as(old)
        assert torch.equal(a,mask) and torch.equal(b,1-mask)
    selected={}
    for mode,arm in [('joint','full'),('point','point'),('skeleton','skeleton')]:
        rr,_=rows('tlp',arm);rr=[r for r in rr if r['split']=='train']
        samples=[min(rr,key=lambda r:r['query_nodes']),max(rr,key=lambda r:r['query_nodes']),
                 max(rr,key=lambda r:len(load(r['path'])['g']['pf']))]
        selected[mode]=list({r['key']:r for r in samples}.values())
    for j in p['jobs']:
        a,d=j['arm'],j['descriptor'];setup();m=Model('tlp',a,d).cuda()
        parent_i=json.loads((parent_result(a,d)/'identity.json').read_text())
        assert state_digest(m)==parent_i['initial_state_sha256']
        measurements=[]
        for r in selected[modal(a)]:
            item=load(r['path']);g=move(item['g']);y=item['target'].cuda();obs=item['observed'].cuda();ref=None
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(4):
                m.zero_grad(set_to_none=True);z=m(g);loss=missing_edge_loss(z,y,obs);loss.backward()
                grads={n:v.grad.detach().cpu().clone() for n,v in m.named_parameters() if v.grad is not None}
                assert all(torch.isfinite(v).all() for v in grads.values())
                if ref is None:ref=(z.detach().cpu(),grads)
                else:
                    assert torch.equal(z.detach().cpu(),ref[0])
                    assert grads.keys()==ref[1].keys() and all(torch.equal(v,ref[1][n]) for n,v in grads.items())
                del z,loss
            assert any(n.startswith('scale.') and v.abs().sum()>0 for n,v in grads.items())==(a!='no_scale')
            measurements.append(dict(key=r['key'],query_nodes=len(y),points=len(g['pf']),repeats=4,exact=True,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del g,y,obs,ref,grads
        results[job_key(a,d)]=dict(parameters=sum(v.numel() for v in m.parameters()),initial_state_sha256=state_digest(m),measurements=measurements)
        print(job_key(a,d),measurements,flush=True);del m;torch.cuda.empty_cache()
    # Exact endpoint equivalence at both initial and trained parent weights.
    endpoint_checks=[]
    for d in ('voxel','neighborhood'):
        for a in ARMS:
            setup();m=Model('tlp',a,d).cuda();base=ParentModel('tlp',a,d).cuda();base.load_state_dict(m.state_dict())
            for weights in ['initial','final']:
                if weights=='final':
                    state=torch.load(parent_result(a,d)/'final.pt',map_location='cpu',weights_only=False)['model']
                    m.load_state_dict(state);base.load_state_dict(state)
                for r in selected[modal(a)]:
                    g=move(load(r['path'])['g'])
                    with torch.no_grad():assert torch.equal(m(g),base(g)),(a,d,weights,r['key'])
                endpoint_checks.append(dict(arm=a,descriptor=d,weights=weights,exact=True))
            del m,base;torch.cuda.empty_cache()
    setup();m=Model('tlp','skeleton','voxel').cuda()
    for r in selected['skeleton']:
        g=move(load(r['path'])['g']);reference=None
        for d in DESCRIPTORS:
            m.factor_descriptor=d
            with torch.no_grad():z=m(g)
            if reference is None:reference=z
            else:assert torch.equal(z,reference)
    hashes=0
    for a in ('full','point','skeleton'):
        rr,_=rows('tlp',a)
        for r in rr:assert sha(r['path'])==r['sha256'];hashes+=1
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),results=results,
        endpoint_checks=endpoint_checks,cache_hashes=hashes,seconds=time.time()-begin,test_used=False,
        tests=['exact_channel_selection','channel_selection_gradients','unchanged_initialization_and_parameters',
               'four_exact_forward_backward_repeats','scale_encoding_retained','initial_and_final_parent_endpoint_exact',
               'skeleton_descriptor_independence','all_cache_hashes']))
    print('contracts passed',flush=True)

if __name__=='__main__':main()

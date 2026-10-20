import json,time
import torch
from .common import *
from .model import Model,weighted_region_sum
from .train import state_digest
from tools.cpsf_point_geometry.model import Model as Native
from tools.cpsf_missing_edge.objective import missing_edge_loss

def primitive():
    # Dense reference checks formula and derivative, including repeated indices.
    row=torch.tensor([0,0,1,1,1,3],device='cuda')
    col=torch.tensor([0,1,1,1,3,2],device='cuda')
    w=torch.tensor([.3,.7,.2,.1,.7,1.],device='cuda',dtype=torch.float64)
    x=torch.randn(4,5,device='cuda',dtype=torch.float64,requires_grad=True)
    dense=torch.zeros(4,4,dtype=x.dtype,device=x.device)
    for i in range(len(row)):dense[row[i],col[i]]+=w[i]
    z=weighted_region_sum(x,row,col,w,4);target=dense@x
    assert torch.allclose(z,target,atol=1e-12,rtol=1e-12)
    upstream=torch.randn_like(z)
    grad=torch.autograd.grad(z,x,upstream)[0]
    assert torch.allclose(grad,dense.T@upstream,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda v:weighted_region_sum(v,row,col,w,4),(x,),atol=1e-6)
    empty=weighted_region_sum(x,row[:0],col[:0],w[:0],4)
    assert not empty.any() and not torch.autograd.grad(empty.sum(),x)[0].any()

def main():
    setup();p=verify();primitive();results={};begin=time.time()
    for job in p['jobs']:
        arm,d=job['arm'],job['descriptor'];setup();m=Model('tlp',arm,d).cuda()
        identity=json.loads((PARENT/'results'/('tlp_'+parent_arm(arm))/'identity.json').read_text())
        assert state_digest(m)==identity['initial_state_sha256']
        rr,_=rows('tlp',arm);rr=[r for r in rr if r['split']=='train']
        selected={r['key']:r for r in [min(rr,key=lambda r:r['query_nodes']),max(rr,key=lambda r:r['query_nodes'])]}
        # Probe true maximum point count too; it need not be max query count.
        largest=max(rr,key=lambda r:len(load(r['path'])['g']['pf']))
        selected[largest['key']]=largest;checks=[]
        native=Native('tlp',arm).cuda();native.use_descriptor=d=='neighborhood';native.load_state_dict(m.state_dict())
        for r in selected.values():
            item=load(r['path']);g=move(item['g']);y=item['target'].cuda();obs=item['observed'].cuda()
            with torch.no_grad():
                old=native(g);new=m(g)
                assert torch.allclose(new,old,atol=2e-6,rtol=2e-6)
                delta=float((new-old).abs().max())
            ref=None;torch.cuda.reset_peak_memory_stats()
            for repeat in range(4):
                m.zero_grad(set_to_none=True);z=m(g);loss=missing_edge_loss(z,y,obs);loss.backward()
                grads={n:v.grad.detach().cpu().clone() for n,v in m.named_parameters() if v.grad is not None}
                out=z.detach().cpu()
                assert all(torch.isfinite(v).all() for v in grads.values())
                if ref is None:ref=(out,grads)
                else:
                    assert torch.equal(out,ref[0]),('output drift',arm,d,r['key'])
                    assert all(torch.equal(v,ref[1][n]) for n,v in grads.items()),('gradient drift',arm,d,r['key'])
                del z,loss
            checks.append(dict(key=r['key'],query_nodes=len(y),points=len(g['pf']),repeats=4,
                               output_and_all_gradients_exact=True,native_output_max_delta=delta,
                               peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del g,y,obs,ref,grads
        results[job_key(arm,d)]=dict(parameters=sum(v.numel() for v in m.parameters()),initial_state_sha256=state_digest(m),checks=checks)
        print(job_key(arm,d),checks,flush=True);del m,native;torch.cuda.empty_cache()
    counts={}
    for arm in ['full','point','skeleton']:
        rr,_=rows('tlp',arm)
        for r in rr:assert sha(r['path'])==r['sha256']
        counts[modal(arm)]=len(rr)
    save(ROOT/'contracts/tlp.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),results=results,
        cache_counts=counts,seconds=time.time()-begin,test_used=False,
        tests=['dense_formula_and_adjoint','float64_gradcheck','empty_support',
               'matched_initialization','same_parameter_count','native_forward_agreement',
               'four_exact_forward_and_backward_repeats_on_small_max_nodes_max_points',
               'all_original_cache_hashes']))
    print('contracts passed',flush=True)

if __name__=='__main__':main()

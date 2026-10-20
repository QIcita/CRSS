"""Mechanism contracts plus full observed maximum-shape forward/backward."""
import argparse
import gc
import pickle
import time
from .common import *
from .data import load_item,move
from .model import Model,PathMessage,OrdinaryAttention,VARIANTS
from tools.cpsf_raw_point_attention.model import Model as Parent


def synthetic():
    from tools.cpsf_v21_segment_units.model import add
    m=PathMessage().cuda();torch.manual_seed(42)
    graph=torch.randn(2,96,device='cuda',requires_grad=True)
    patch=torch.randn(5,96,device='cuda',requires_grad=True)
    g=dict(nregions=2,pr=torch.tensor([0,0,0,1,1],device='cuda'),pj=torch.arange(5,device='cuda'),
           pw=torch.ones(5,device='cuda'),patch_mass=torch.tensor([1.,2.,1.,3.,2.],device='cuda'),
           surface_mass=torch.tensor([4.,5.],device='cuda'),phi=torch.randn(5,5,device='cuda'))
    with torch.no_grad():m.output.weight.normal_(0,.1)
    a=m(graph,patch,g,'path_message');b=m(-graph,patch,g,'path_message')
    assert not torch.allclose(a,b),'graph must affect messages'
    assert torch.equal(m(graph,patch,g,'constant_message'),m(-graph,patch,g,'constant_message'))
    a.square().sum().backward();assert graph.grad.abs().sum()>0 and patch.grad.abs().sum()>0
    uniform=patch[0].expand_as(patch)
    assert m(graph,uniform,g,'path_message').abs().max()<2e-6
    p=torch.tensor([4,2,0,3,1],device='cuda');changed=dict(g,pr=g['pr'][p],pj=g['pj'][p],pw=g['pw'][p],phi=g['phi'][p])
    assert torch.allclose(a,m(graph,patch,changed,'path_message'),atol=2e-6,rtol=2e-5)
    # Duplicate an observation and split its mass, without changing total mass.
    gg=dict(g,pr=torch.cat((g['pr'],g['pr'][:1])),pj=torch.arange(6,device='cuda'),
            pw=torch.ones(6,device='cuda'),patch_mass=torch.tensor([.5,2.,1.,3.,2.,.5],device='cuda'),phi=torch.cat((g['phi'],g['phi'][:1])))
    pp=torch.cat((patch,patch[:1]));assert torch.allclose(a,m(graph,pp,gg,'path_message'),atol=2e-6,rtol=2e-5)
    old=m.chunk;m.chunk=2;c=m(graph,patch,g,'path_message');m.chunk=old
    assert torch.allclose(a,c,atol=2e-6,rtol=2e-5)
    # Exact checkpoint/direct gradient equality for a fixed chunk decomposition.
    m.checkpointed=False;a=m(graph,patch,g,'path_message');da=torch.autograd.grad(a.sum(),(graph,patch))
    m.checkpointed=True;b=m(graph,patch,g,'path_message');db=torch.autograd.grad(b.sum(),(graph,patch))
    assert torch.equal(a,b) and all(torch.equal(x,y) for x,y in zip(da,db))
    return dict(graph_condition_changes_messages=True,constant_arm_independent_of_graph=True,
                constant_values_zero=True,mass_split_invariant=True,incidence_permutation_invariant=True,
                direct_checkpoint_output_and_gradients_exact=True,chunk_difference_tolerated_at_2e_6=True)


def grouping_check(g):
    edges=[]
    for lo,hi,left,right,width in g['att_layout']:
        ix=g['att_indices'][left:right].reshape(hi-lo,width);valid=g['att_valid'][left:right].reshape_as(ix)
        ids=g['att_regions'][lo:hi]
        assert torch.equal(g['pr'][ix][valid],ids[:,None].expand_as(ix)[valid])
        edges.append(ix[valid])
    got=torch.cat(edges) if edges else torch.empty(0,dtype=torch.long)
    keep=g['has_g'][g['pr']]&g['has_p'][g['pr']]
    assert torch.equal(got.sort().values,torch.where(keep)[0])


def main(task):
    setup();p=verify();rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
    assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
    chosen=list({r['key']:r for r in [min(tr,key=lambda r:r['points'])]+[max(tr,key=lambda r:r[k]) for k in ('nodes','regions','patches','points','incidences')]}.values())
    for row in chosen:
        rng=pickle.dumps(np.random.get_state());ts=torch.get_rng_state().clone();g=load_item(row)
        assert pickle.dumps(np.random.get_state())==rng and torch.equal(ts,torch.get_rng_state())
        grouping_check(g);assert 'target' not in g and 'label' not in g
    setup();old=Parent(task,'r9_lpa_attention').cuda().eval();setup();m=Model(task,'path_message').cuda().eval()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.state_dict().items())
    # Nonzero existing SPoTr projection prevents fallback from hiding a capture bug.
    with torch.no_grad():old.raw_refine.output.weight.copy_(torch.eye(96,device='cuda')*.01);m.raw_refine.output.weight.copy_(old.raw_refine.output.weight)
    g=move(load_item(min(tr,key=lambda r:r['points'])))
    with torch.no_grad():assert torch.equal(old(g),m(g))
    del old,m,g;gc.collect();torch.cuda.empty_cache()
    report=dict(task=task,passed=False,protocol_sha256=sha(ROOT/'protocol.json'),synthetic=synthetic(),
                parent_shared_states_exact=True,parent_fallback_with_active_lpa_exact=True,
                original_manifest_and_data_hashes_verified=True,input_rng_unchanged=True,cell_disjoint=True,full_attention_incidence_coverage=True,
                variants={},test_used=False)
    for kind in VARIANTS:
        setup();m=Model(task,kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);active=set()
        warm=min(tr,key=lambda r:r['points'])
        for _ in range(4):
            opt.zero_grad(set_to_none=True);z=m(move(load_item(warm)));z.square().sum().backward()
            active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        cases=[]
        for row in chosen:
            begin=time.time();g=move(load_item(row));m.train();m.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            z=m(g);z.square().sum().backward();torch.cuda.synchronize()
            assert torch.isfinite(z).all() and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            m.eval()
            with torch.no_grad():
                before=state_sha(m);a=m(g);b=m(g);assert torch.equal(a,b) and state_sha(m)==before
                m.message_enabled=False;c=m(g);m.message_enabled=True
                assert torch.isfinite(c).all()
            cases.append(dict(key=row['key'],nodes=row['nodes'],regions=row['regions'],patches=row['patches'],points=row['points'],incidences=row['incidences'],seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            print(kind,task,cases[-1],flush=True);del g,z,a,b,c;gc.collect();torch.cuda.empty_cache()
        for prefix in ('graph.','point.','position.','scale.','joint.','processor.','native_point.encoder.','raw_refine.local.attn_local.','raw_refine.output.','region_message.output.','region_message.geometry.'):
            assert any(n.startswith(prefix) for n in active),(task,kind,prefix)
        for n,v in m.region_message.named_parameters():assert 'region_message.'+n in active,('inactive new parameter',task,kind,n)
        dev=max(va,key=lambda r:r['incidences'])
        with torch.no_grad():assert torch.isfinite(m(move(load_item(dev)))).all()
        report['variants'][kind]=dict(cases=cases,active_gradients=sorted(active),parameters=sum(v.numel() for v in m.parameters()),largest_dev_unlabeled_forward=dev['key'])
        del m,opt;gc.collect();torch.cuda.empty_cache()
    report['passed']=True;save(ROOT/'acceptance'/f'{task}.json',report)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True);main(a.parse_args().task)

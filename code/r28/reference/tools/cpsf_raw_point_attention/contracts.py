"""Official forward/gradient, lossless grouping, R9 fallback and maximum-shape gates."""
import argparse
import gc
import inspect
import pickle
import time
from .common import *
from .data import grouping, load_item, move
from .model import Model, RawPointRefinement, VARIANTS
from .official import symbols
from tools.cpsf_utonia_context.model import Model as OriginalR9


def check_grouping(g):
    eligible = g['raw_counts'] > 1
    assert torch.equal(torch.sort(g['raw_patch_ids']).values, torch.where(eligible)[0])
    seen = []
    for lo, hi, left, right, width in g['raw_layout']:
        ids = g['raw_patch_ids'][lo:hi]
        ix = g['raw_indices'][left:right].reshape(hi-lo, width)
        valid = g['raw_valid'][left:right].reshape(hi-lo, width)
        assert torch.equal(valid.sum(1), g['raw_counts'][ids])
        assert torch.equal(g['point_owner'][ix][valid], ids[:, None].expand_as(ix)[valid])
        seen.append(ix[valid])
    want = torch.where(eligible[g['point_owner']])[0]
    actual = torch.sort(torch.cat(seen)).values if seen else want.new_empty(0)
    assert torch.equal(actual, want)


def synthetic():
    native = symbols()['new_attention']
    z = torch.randn(1, 96, 7, 5, device='cuda', requires_grad=True)
    logit = torch.randn_like(z, requires_grad=True)
    actual = native(z, attn=logit)
    expected = (z * torch.softmax(logit, -1)).sum(-1)
    assert torch.equal(actual, expected)
    grad = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (z, logit), grad, retain_graph=True)
    b = torch.autograd.grad(expected, (z, logit), grad)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    owner = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    g = dict(point_owner=owner, point_weight=torch.tensor([1., 2., 3., 1., 2., 1., 1., 1., 1., 1.]),
             patch_mass=torch.tensor([6., 1., 3., 4.]), patches=4,
             pf=torch.randn(10, 4), raw_patch_pos=torch.randn(4, 3))
    g.update(grouping(owner.numpy(), 4));check_grouping(g);g=move(g)
    layer = RawPointRefinement().cuda()
    x = torch.randn(10, 96, device='cuda', requires_grad=True)
    official_path = str(ROOT / 'vendor/SPoTr/openpoints/models/backbone/spotr.py')
    assert layer.local.forward.__func__.__code__.co_filename == official_path
    assert symbols()['new_attention'].__code__.co_filename == official_path
    assert 'new_attention(self.convs(fj), attn = self.attn_local(fj))' in inspect.getsource(layer.local.forward)
    norms = [v for v in layer.local.modules() if isinstance(v, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm))]
    assert not norms
    for kind in VARIANTS:
        a = layer.descriptors(x, g, kind, checkpointed=False)
        assert torch.equal(a[1], torch.zeros_like(a[1]))
        grad = torch.randn_like(a)
        da = torch.autograd.grad(a, x, grad, retain_graph=True)[0]
        b = layer.descriptors(x, g, kind, checkpointed=True)
        db = torch.autograd.grad(b, x, grad)[0]
        assert torch.equal(a,b) and torch.equal(da,db)
        # Change only padding contents, not actual members.
        changed=dict(g);changed['raw_indices']=g['raw_indices'].clone()
        changed['raw_indices'][~g['raw_valid']]=len(x)-1
        assert torch.equal(a,layer.descriptors(x,changed,kind,checkpointed=False))
        # Different exact grouping budget cannot change per-patch features.
        tiny=dict(g);tiny.update({k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in grouping(owner.numpy(),4,slot_budget=4).items()})
        assert torch.allclose(a,layer.descriptors(x,tiny,kind,checkpointed=False),atol=2e-6,rtol=2e-6)
        layer.train();a=layer.descriptors(x,g,kind,False)
        layer.eval();b=layer.descriptors(x,g,kind,False)
        assert torch.equal(a,b)
    # Uniform logit prior computes precisely original mass-weighted average.
    weights=torch.tensor([1.,2.,3.],device='cuda')
    values=torch.randn(1,96,1,3,device='cuda')
    a=native(values,attn=weights.log()[None,None,None].expand_as(values))
    b=(values*weights[None,None,None]).sum(-1)/weights.sum()
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    return dict(official_attention_forward_and_gradient_exact=True,official_lpa_forward_source_unchanged=True,
                original_mass_prior_correct=True,all_original_members_preserved=True,
                singleton_exact_fallback=True,padding_invariant=True,group_budget_invariant=True,
                checkpoint_outputs_and_gradients_exact=True,same_sample_train_eval_exact=True)


def main(task):
    setup();p=verify();rr=rows(task);tr=[r for r in rr if r['split']=='train']
    assert sha(ROOT/'data'/task/'manifest.json')==sha(PARENT/'data'/task/'manifest.json')
    for row in rr:
        assert sha(row['path'])==row['sha256'] and sha(row['feature_path'])==row['feature_sha256']
    assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in rr if r['split']=='val')
    chosen=list({r['key']:r for r in [min(tr,key=lambda r:r['points'])]+[max(tr,key=lambda r:r[k]) for k in ('nodes','regions','patches','points','incidences')]}.values())
    for row in chosen:
        state=pickle.dumps(np.random.get_state());rng=torch.get_rng_state().clone()
        g=load_item(row);check_grouping(g)
        assert state==pickle.dumps(np.random.get_state()) and torch.equal(rng,torch.get_rng_state())
        assert 'target' not in g and 'label' not in g
    setup();old=OriginalR9(task,'patch_rms_010').cuda().eval()
    setup();m=Model(task,VARIANTS[0]).cuda().eval()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.state_dict().items())
    g=move(load_item(max(tr,key=lambda r:r['points'])))
    with torch.no_grad():
        y=old(g)
        for kind in VARIANTS:
            m.kind=kind;assert torch.equal(y,m(g)),('initial fallback',kind)
    added=sum(v.numel() for v in m.raw_refine.parameters());trainable=sum(v.numel() for v in m.raw_refine.parameters() if v.requires_grad)
    del old,m,g;gc.collect();torch.cuda.empty_cache()
    report=dict(task=task,passed=False,protocol_sha256=sha(ROOT/'protocol.json'),
        original_manifest_and_data_hashes_valid=True,train_dev_cell_disjoint=True,
        original_r9_initialization_and_output_exact=True,input_loading_rng_unchanged=True,
        synthetic=synthetic(),variants={},additional_parameters=added,trainable_added_parameters=trainable,test_used=False)
    for kind in VARIANTS:
        setup();m=Model(task,kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
        active=set();cases=[];warm=max(tr,key=lambda r:r['points'])
        calls=[]
        hook=m.raw_refine.local.register_forward_hook(lambda module,args,out:calls.append(1))
        for _ in range(2):
            m.train();opt.zero_grad(set_to_none=True);z=m(move(load_item(warm)))
            z.square().sum().backward();active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        for row in chosen:
            begin=time.time();g=move(load_item(row));m.train();m.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats();z=m(g);z.square().sum().backward();torch.cuda.synchronize()
            assert torch.isfinite(z).all() and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
            active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
            m.eval()
            with torch.no_grad():assert torch.equal(m(g),m(g))
            cases.append(dict(key=row['key'],points=row['points'],patches=row['patches'],incidences=row['incidences'],seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            print(kind,task,cases[-1],flush=True)
            del g,z;gc.collect();torch.cuda.empty_cache()
        assert calls;hook.remove()
        for prefix in ('graph.','point.','position.','scale.','joint.','processor.','native_point.encoder.','native_point.proj.','raw_refine.local.convs.','raw_refine.local.ffn.','raw_refine.output.'):
            assert any(n.startswith(prefix) for n in active),(kind,task,prefix)
        has_attention=any(n.startswith('raw_refine.local.attn_local.') for n in active)
        assert has_attention==(kind=='r9_lpa_attention')
        assert m.raw_refine.local.alpha.grad is None
        devmax=max((r for r in rr if r['split']=='val'),key=lambda r:r['incidences'])
        with torch.no_grad():assert torch.isfinite(m(move(load_item(devmax)))).all()
        report['variants'][kind]=dict(cases=cases,active_gradients=sorted(active),official_lpa_calls=len(calls),largest_dev_forward_only=devmax['key'],parameters=sum(v.numel() for v in m.parameters()))
        del m,opt;gc.collect();torch.cuda.empty_cache()
    report['passed']=True;save(ROOT/'acceptance'/f'{task}.json',report)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True);main(a.parse_args().task)

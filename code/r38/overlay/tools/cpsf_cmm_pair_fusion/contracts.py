"""Algebra, finite derivatives, checkpoint and bounded-fusion checks."""
import argparse
import itertools
import torch
from .common import ROOT, save
from .operator import PairFusion, PAIRS, WEIGHTS


def run(device):
    torch.set_num_threads(2); torch.manual_seed(42)
    dtype = torch.float64 if device == 'cpu' else torch.float32
    tol = 1e-12 if device == 'cpu' else 2e-6
    op = PairFusion().to(device=device, dtype=dtype)
    op.checkpointed = False
    xs = tuple(torch.randn(3,96,device=device,dtype=dtype,requires_grad=True) for _ in range(3))
    z = tuple(torch.zeros_like(x) for x in xs)
    base = xs[0] + (.1*xs[2] + .1*xs[1])
    initial = base + op(*xs)
    assert torch.equal(initial,base)
    ga=torch.autograd.grad(initial.sum(),xs,retain_graph=True)
    gb=torch.autograd.grad(base.sum(),xs)
    assert all(torch.equal(a,b) for a,b in zip(ga,gb))
    with torch.no_grad():
        for b in op.outputs:b.normal_(0,.1)
    terms=op.terms(*xs); val=op(*xs)
    bound=op.alpha*sum(w*x.norm(dim=-1) for x,w in zip(xs,WEIGHTS))
    assert (val.norm(dim=-1)<=bound+tol).all()
    errors=[]
    for k,(i,j) in enumerate(PAIRS):
        for missing in (i,j):
            args=list(xs);args[missing]=z[missing]
            assert torch.equal(op.terms(*args)[k],z[0])
        args=list(z);args[i]=xs[i];args[j]=xs[j]
        errors.append(float((op(*args)-terms[k]).abs().max()))
        assert errors[-1]<tol
    third=torch.zeros_like(xs[0])
    for mask in itertools.product((0,1),repeat=3):
        args=[x if m else zero for x,zero,m in zip(xs,z,mask)]
        third=third+(-1)**(3-sum(mask))*op(*args)
    assert float(third.abs().max())<tol
    if device=='cpu':
        params=dict(op.named_parameters())
        assert torch.autograd.gradcheck(lambda *a:torch.func.functional_call(
            op,dict(zip(params,a[3:])),a[:3]),xs+tuple(params.values()),fast_mode=True)
    target=torch.randn_like(val)
    (val*target).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and (p.grad!=0).any() for p in op.parameters())
    cases=[]
    for n in (1,17,4097):
        inputs=tuple(torch.randn(n,96,device=device,dtype=dtype,requires_grad=True) for _ in range(3))
        target=torch.randn_like(inputs[0]);params=list(op.parameters())
        op.checkpointed=False;a=op(*inputs);g0=torch.autograd.grad((a*target).sum(),list(inputs)+params)
        op.checkpointed=True;b=op(*inputs);g1=torch.autograd.grad((b*target).sum(),list(inputs)+params)
        assert torch.equal(a,b)
        for v,w in zip(g0,g1):torch.testing.assert_close(v,w,atol=tol,rtol=tol)
        cases.append(dict(patches=n,output_exact=True,gradient_max_abs=max(float((v-w).abs().max()) for v,w in zip(g0,g1))))
    report=dict(passed=True,device=device,dtype=str(dtype),parameters=sum(p.numel() for p in op.parameters()),
        zero_output_and_input_gradient_exact=True,pair_zero_anchor_exact=True,perturbation_bound=True,
        anchored_pair_errors=errors,third_order_contrast_max_abs=float(third.abs().max()),
        input_and_all_parameter_gradcheck=device=='cpu',all_parameters_active=True,checkpoint_cases=cases,
        test_used=False,real_model_test=False)
    save(ROOT/('operator_contract_'+device+'.json'),report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),required=True)
    run(p.parse_args().device)

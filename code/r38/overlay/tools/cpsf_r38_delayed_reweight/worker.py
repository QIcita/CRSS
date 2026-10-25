import argparse
import fcntl
import gc
import os
import time
import traceback
import numpy as np
import torch
from .common import ROOT, DENSE, TASK, LABELS, read, save, sha, rows, verify
from .backend import Backend, DenseBackend, digest
from . import objective
from tools.cpsf_cmm_pair_fusion import worker as inherited

def cpu_contract(training):
    objective.configure(training)
    z = torch.tensor([[.4, -.7, .9, -.1, .5, -.9], [1., -.3, .4, -.2, .5, .1]], dtype=torch.float64)
    trace = []
    for epoch in (1, 30, 31, 32, 33, 34, 35, 36, 50):
        objective.set_epoch(epoch)
        for label in range(6):
            x = z[0].clone().requires_grad_()
            base = objective.original_loss(x, label, TASK, 1.)
            new = objective.loss_for(x, label, TASK, 1.)
            gb = torch.autograd.grad(base, x, retain_graph=True)[0]
            gn = torch.autograd.grad(new, x)[0]
            w = objective.weights(epoch)[label]
            torch.testing.assert_close(new, base*w, rtol=0, atol=1e-15)
            torch.testing.assert_close(gn, gb*w, rtol=1e-14, atol=1e-15)
            if epoch <= 30:
                assert torch.equal(base, new) and torch.equal(gb, gn)
        trace.append(dict(epoch=epoch, weights=objective.weights(epoch)))
    # Exact microbatch semantics: per-cell accumulation equals batch loss / cell count.
    for count in (1, 2):
        x = z[:count].clone().requires_grad_(); labels = torch.tensor([4,2][:count])
        ce = torch.nn.functional.cross_entropy(x, labels, label_smoothing=.05, reduction='none')
        expected = (ce*torch.tensor([objective.weights(50)[int(y)] for y in labels], dtype=x.dtype)).mean()
        actual = sum(objective.loss_for(x[i], int(labels[i]), TASK, 1.) for i in range(count))/count
        torch.testing.assert_close(expected, actual, rtol=1e-14, atol=1e-15)
        ga = torch.autograd.grad(actual, x, retain_graph=True)[0]; ge = torch.autograd.grad(expected, x)[0]
        torch.testing.assert_close(ga, ge, rtol=1e-14, atol=1e-15)
    objective.set_epoch(1)
    return dict(passed=True, epochs=trace, all_six_classes=True, gradient_scaling_verified=True,
                single_and_multi_cell_accumulation=True, first30_loss_and_gradient_exact=True)

def compatibility(training, cfg):
    old = DenseBackend('r38', TASK, training); new = Backend('r38', TASK, training)
    initial = digest(new.model)
    assert initial == digest(old.model) == read(DENSE/'acceptance/r38_flywire_coarse.json')['initial_state_sha256']
    row = min(training, key=lambda r:r['points'])
    g, y, _ = new.item(row)
    old.model.eval(); new.model.eval()
    with torch.no_grad(): assert torch.equal(old.model(g), new.model(g))
    opt0, _ = inherited.optimizer(old, cfg); opt1, _ = inherited.optimizer(new, cfg)
    trace = []
    for step in range(6):
        epoch = 1 if step < 3 else 30
        new.set_epoch(epoch, 50); values = []
        for b, opt, loss_fn in ((old,opt0,objective.original_loss),(new,opt1,objective.loss_for)):
            torch.manual_seed(42+step); b.model.train(); opt.zero_grad(set_to_none=True)
            z = b.model(g); loss = loss_fn(z, y, TASK, 1.); loss.backward()
            torch.nn.utils.clip_grad_norm_(b.model.parameters(), 5, error_if_nonfinite=True)
            opt.step(); values.append(float(loss.detach()))
        assert values[0] == values[1] and digest(old.model) == digest(new.model)
        trace.append(dict(step=step+1, epoch=epoch, loss=values[0], all_parameters_exact=True))
    del old, new, opt0, opt1, g, z, loss; gc.collect(); torch.cuda.empty_cache()
    return dict(passed=True, initial=initial, exact_parent_forward=True, six_updates=trace)

def acceptance():
    folder = ROOT/'acceptance'; folder.mkdir(exist_ok=True)
    lock = open(folder/'r38_flywire_coarse.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    p = verify(); training = [r for r in rows(TASK) if r['split']=='train']; cfg = p['tasks'][TASK]
    start = time.time(); progress = folder/'progress.json'
    save(progress, dict(state='cpu_loss_contract', pid=os.getpid()))
    cpu = cpu_contract(training)
    save(progress, dict(state='parent_exact_updates', pid=os.getpid()))
    control = compatibility(training, cfg)
    b = Backend('r38', TASK, training); m = b.model; initial = digest(m)
    assert initial == control['initial']
    opt, _ = inherited.optimizer(b, cfg); active = set(); trace = []
    torch.cuda.reset_peak_memory_stats()
    warm = min(training, key=lambda r:r['points'])
    for step in range(6):
        b.set_epoch(35,50); m.train(); opt.zero_grad(set_to_none=True)
        g,y,_ = b.item(warm); loss = objective.loss_for(m(g),y,TASK,1.); loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(step=step+1, loss=float(loss.detach()), weight=objective.weights(35)[int(y)]))
    dims=['nodes','points','patches','regions','incidences']
    chosen=[min([r for r in training if r['target']==c],key=lambda r:r['points']) for c in range(6)]
    chosen += [max(training,key=lambda r:r[k]) for k in dims]
    chosen=list({r['key']:r for r in chosen}.values());cases=[]
    for row in chosen:
        save(progress,dict(state='weighted_real_backward',completed=len(cases),total=len(chosen),key=row['key'],pid=os.getpid()))
        b.set_epoch(50,50);m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(row)
        z=m(g);base=objective.original_loss(z,y,TASK,1.);loss=objective.loss_for(z,y,TASK,1.)
        torch.testing.assert_close(loss,base*objective.weights(50)[int(y)],rtol=0,atol=0)
        loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
        active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
        cases.append(dict(key=row['key'],target=int(y),weight=objective.weights(50)[int(y)],
                          base_loss=float(base.detach()),weighted_loss=float(loss.detach()),**{k:row[k] for k in dims}))
        del g,y,z,base,loss;gc.collect();torch.cuda.empty_cache()
    for prefix in ['native_point.encoder','graph','point','joint','raw_refine.output','region_message','cmm_pair_fusion']:
        assert any(n.startswith(prefix) for n in active),prefix
    assert {n for n,_ in m.named_parameters() if n.startswith('cmm_pair_fusion.')}<=active
    m.eval();before=digest(m)
    with torch.no_grad():
        g,_,_=b.item(warm);a=b.eval_forward(g).cpu()
        g,_,_=b.item(chosen[-1]);b.eval_forward(g)
        g,_,_=b.item(warm);c=b.eval_forward(g).cpu()
    assert torch.equal(a,c) and digest(m)==before
    result=dict(passed=True,model='r38',task=TASK,protocol_sha256=sha(ROOT/'protocol.json'),
                manifest_sha256=p['input_manifests'][TASK],initial_state_sha256=initial,
                official_sources=b.sources,cpu_contract=cpu,compatibility=control,updates=trace,cases=cases,
                active_parameters=sorted(active),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                parent_architecture_acceptance=str(DENSE/'acceptance/r38_flywire_coarse.json'),
                eval_state_unchanged=True,input_isolation_exact=True,test_used=False,pretrained=False,
                training_weights_reused=False,seconds=time.time()-start)
    verify();save(folder/'r38_flywire_coarse.json',result);save(progress,dict(state='passed',seconds=time.time()-start,pid=os.getpid()))
    print('ACCEPTED',result['seconds'],result['peak_gpu_bytes'],flush=True)

parent_metrics = inherited.metrics
def metrics(y,z,task):
    result=parent_metrics(y,z,task);cm=np.asarray(result['confusion_matrix']);tp=np.diag(cm)
    support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(6),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(6),where=support!=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(6),where=(support+predicted)!=0)
    result['weighted_f1']=float((f1*support).sum()/support.sum())
    result['per_class']=[dict(label=i,name=LABELS[i],support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i in range(6)]
    return result

def train():
    inherited.ROOT=ROOT;inherited.rows=rows;inherited.verify=verify;inherited.Backend=Backend
    inherited.loss_for=objective.loss_for;inherited.metrics=metrics
    objective.RECORD_STATE=True
    inherited.train('r38',TASK)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--accept',action='store_true');args=parser.parse_args()
    try: acceptance() if args.accept else train()
    except BaseException as error:
        save(ROOT/('acceptance' if args.accept else 'results')/'r38_flywire_coarse_failure.json',
             dict(error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
        raise

"""H01 only. Candidate processes can evaluate inner validation only."""
import argparse,time
from .io import *
from .model import Model,ARMS
from torch import nn
ROOT=Path(__file__).resolve().parents[2]

def validated_protocol():
    p=json.loads((ROOT/'search_protocol.json').read_text())
    assert sha(ROOT/'split_manifest.json')==p['split_manifest_sha256']
    assert {x.name:sha(x) for x in Path(__file__).parent.glob('*.py')}==p['source_sha256']
    assert sha(ROOT/'CPSF_V21_SEGMENT_UNITS_PROTOCOL_20260930.md')==p['protocol_sha256']
    return p

def items(rows,cache,arm):
    def prepare_item(r):
        item=load(r['path'])
        if arm not in ('no_fusion','point','skeleton'):
            from .segments import build_units
            item['geometry']['units_node' if arm=='node_units' else 'units_segment']=build_units(item['geometry'],node_units=arm=='node_units')
        return item
    data=[cache.get(r['sample'],lambda r=r:prepare_item(r)) for r in rows]
    return [r['geometry'] for r in data],[r['query'] for r in data]

@torch.no_grad()
def evaluate(model,rows,cache):
    model.eval();records=[];ys=[];preds=[]
    for lo in range(0,len(rows),4):
        sub=rows[lo:lo+4];gs,qs=items(sub,cache,model.arm);z=model(gs,qs).cpu();p=z.argmax(1)
        ys.extend(r['target'] for r in sub);preds.extend(p.tolist())
        records.extend(dict(sample=r['sample'],cell=r['cell'],target=r['target'],prediction=int(t),logits=v) for r,t,v in zip(sub,p,z.tolist()))
    return dict(**classification(ys,preds),samples=len(rows),cells=len({r['cell'] for r in rows}),rows=records)

def run(a):
    setup();p=validated_protocol();split=json.loads((ROOT/'split_manifest.json').read_text())['rows']
    assert not a.out.exists();a.out.mkdir(parents=True)
    if a.phase=='inner':
        cfg=next(c for c in p['candidates'] if c['name']==a.candidate);a.arm=cfg['arm']
        epochs=35;train=split['inner_train'];validation=split['inner_val'];checkpoints=p['checkpoints']
    else:
        selected=json.loads((ROOT/'selection.json').read_text());assert selected['search_protocol_sha256']==sha(ROOT/'search_protocol.json')
        assert selected['gate_passed'];cfg=selected['candidate'];epochs=selected['epoch'];train=split['outer_train'];validation=split['outer_val'];checkpoints=[epochs]
    if a.smoke:
        # Smoke never touches outer validation: one inner training example per label.
        assert a.phase=='inner';train=[next(r for r in train if r['target']==c) for c in range(3)]
        validation=[next(r for r in validation if r['target']==c) for c in range(3)];epochs=2;checkpoints=[2]
    assert not ({r['cell'] for r in train}&{r['cell'] for r in validation})
    for r in train+validation:assert sha(r['path'])==r['sha256']
    model=Model('h01',a.arm,residual_scale=cfg['residual_scale']).cuda()
    cache=Cache(budget=1024*2**20)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=cfg['weight_decay'])
    # Fixed horizon35, also during selected short refit, matches the searched prefix.
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,35,eta_min=.00003)
    protocol=dict(version=p['version'],task='h01',arm=a.arm,phase=a.phase,seed=42,epochs=epochs,smoke=a.smoke,
                  candidate=cfg,model_config=dict(residual_scale=cfg['residual_scale']),
                  search_protocol_sha256=sha(ROOT/'search_protocol.json'),source_sha256=p['source_sha256'],
                  train_cells=sorted({r['cell'] for r in train}),validation_cells=sorted({r['cell'] for r in validation}),
                  registered_parameters=sum(t.numel() for t in model.parameters()),scheduler_horizon=35)
    save(a.out/'protocol.json',protocol);history=[];evaluations={};start=time.time();active_parameters=set()
    for epoch in range(1,epochs+1):
        model.train();begin=time.time();total_loss=0.;seen=0
        order=np.random.permutation(len(train))
        for lo in range(0,len(order),8):
            rows=[train[i] for i in order[lo:lo+8]];opt.zero_grad(set_to_none=True)
            for j in range(0,len(rows),2):
                sub=rows[j:j+2];gs,qs=items(sub,cache,model.arm);logits=model(gs,qs)
                target=torch.tensor([r['target'] for r in sub],device='cuda')
                loss=nn.functional.cross_entropy(logits,target,reduction='sum',label_smoothing=cfg['label_smoothing'])
                assert torch.isfinite(loss);(loss/len(rows)).backward();total_loss+=float(loss.detach());seen+=len(sub)
            nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
        active_parameters.update(name for name,t in model.named_parameters() if t.grad is not None and bool(torch.any(t.grad!=0)))
        schedule.step();record=dict(epoch=epoch,epochs=epochs,seconds=time.time()-begin,mean_ce=total_loss/seen)
        history.append(record);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**record));print(json.dumps(record),flush=True)
        saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,protocol=protocol)
        temp=a.out/'latest.tmp';torch.save(saved,temp);temp.replace(a.out/'latest.pt')
        if epoch in checkpoints:
            e=evaluate(model,validation,cache);evaluations[str(epoch)]=e;save(a.out/'evaluations.json',evaluations)
            torch.save(saved,a.out/f'epoch_{epoch}.pt')
            print('EVAL',a.phase,epoch,e['macro_f1'],flush=True)
    torch.save(saved,a.out/'final.pt')
    if a.phase=='inner':metrics=dict(protocol=protocol,inner_validation=evaluations,seconds=time.time()-start)
    else:metrics=dict(protocol=protocol,val=evaluations[str(epochs)],train=evaluate(model,train,cache),seconds=time.time()-start)
    metrics['nonzero_gradient_parameters_at_epoch_ends']=sum(t.numel() for name,t in model.named_parameters() if name in active_parameters)
    metrics['nonzero_gradient_parameter_names_at_epoch_ends']=sorted(active_parameters)
    save(a.out/'metrics.json',metrics);save(a.out/'status.json',dict(state='complete',epochs=epochs,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inner','refit'],required=True);p.add_argument('--candidate');p.add_argument('--arm',choices=ARMS,default='full');p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise

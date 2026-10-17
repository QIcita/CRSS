"""Verify inherited final predictions and schedule factors before any new run."""
import json,time
import numpy as np
from torch.nn import functional as F
from .common import *
from .train import state_digest
from tools.cpsf_missing_edge.objective import missing_edge_loss

def run():
    p=verify();report=[];lrs={};begin=time.time()
    for task,cfg in p['tasks'].items():
        lrs[task]={}
        for config in ('baseline',)+CONFIGS:
            param=torch.nn.Linear(1,1);opt,sched=optimizer(param,task,config,cfg);sequence=[]
            for epoch in range(cfg['epochs']):
                rate=opt.param_groups[0]['lr'];sequence.append(rate)
                base=cfg['lr']*(.5 if config=='lr_half' else 1)
                expected=base/(1+epoch/(3 if config=='slow_decay' else 1)) if task=='ted' else base*.1**(epoch//(40 if config=='slow_decay' else 20))
                assert abs(rate-expected)<1e-14
                opt.step();sched.step()
            lrs[task][config]=sequence
        for arm,directory in p['baseline_dirs'][task].items():
            setup();root=Path(directory);identity=json.loads((root/'identity.json').read_text())
            model=Model(task,arm).cuda();assert state_digest(model)==identity['initial_state_sha256']
            rr,_=rows(task,arm);train=[r for r in rr if r['split']=='train'];val=[r for r in rr if r['split']=='val']
            assert set(r['group'] for r in train).isdisjoint(r['group'] for r in val)
            assert len(train)==(23967 if task=='ted' else 156) and len(val)==(10762 if task=='ted' else 62)
            ck=torch.load(root/'final.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
            r=min(val,key=lambda r:Path(r['path']).stat().st_size);assert sha(r['path'])==r['sha256']
            item=load(r['path']);g=move(item['g'])
            with torch.no_grad():z=model(g).cpu().numpy()
            if task=='ted':
                with np.load(root/'val_predictions.npz') as saved:
                    matches=np.flatnonzero(saved['keys']==r['key']);assert len(matches)==1
                    assert np.array_equal(z,saved['logits'][matches[0]]),'TED parent output differs'
            else:
                with np.load(root/('prediction_'+r['key']+'.npz')) as saved:
                    assert np.array_equal(torch.from_numpy(z).sigmoid().numpy(),saved['score']),'TLP parent output differs'
            del ck,g,item
            setup();model=Model(task,arm).cuda();model.train()
            r=min(train,key=lambda r:Path(r['path']).stat().st_size);assert sha(r['path'])==r['sha256']
            item=load(r['path']);g=move(item['g']);y=item['target'].cuda();z=model(g)
            loss=F.binary_cross_entropy_with_logits(z,y)*(5 if y[1]>0 else 1) if task=='ted' else missing_edge_loss(z,y,item['observed'].cuda())
            loss.backward();assert torch.isfinite(loss)
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
            report.append(dict(task=task,arm=arm,initial_matches=True,parent_final_prediction_exact=True,finite_gradient=True,train=len(train),val=len(val)))
            del model,item,g,y,z,loss;torch.cuda.empty_cache()
    save(ROOT/'contracts/config.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),checks=report,learning_rates=lrs,seconds=time.time()-begin,test_used=False))
    print('PASS: inherited models, final predictions, initial states, gradients, splits, schedule factors',flush=True)
if __name__=='__main__':run()

"""Recompute development predictions; no test prediction access or score selection."""
import json, hashlib, datetime, subprocess
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

base=Path('/root/megrez-tmp/CPSF_NREDD_BASELINES_20261014')
native=Path('/root/megrez-tmp/CPSF_TLP_NATIVE_OBJECTIVE_20261015')
manifest=json.loads((base/'data/tlp/manifest.json').read_text())
rows=[r for r in manifest['records'] if r['split']=='val']
assert len(rows)==62
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ap=lambda y,s:float(average_precision_score(y,s)) if y.any() else 0.
roots={m:base/'results'/f'{m}_tlp' for m in ['tlpnet','sgformer','polynormer']}
roots['tlpnet_native_objective']=native
r28=Path('/root/megrez-tmp/CPSF_R28_TASK_EXPANSION_20261020_A2/results/r28_lpa_attention_tlp')
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),split='val only',test_predictions_accessed=False,test_metrics_used=False,training_launched=False,methods={},graphs=[])
official=Path('/root/megrez-tmp/NREDD_proj/NREDD')
commit='99afba251f4933bcf11c629df3c1929d9e6e8524'
report['official_source_commit']=commit
report['official_source_parity']={}
for p in (base/'vendor/NREDD').rglob('*.py'):
    rel=str(p.relative_to(base/'vendor/NREDD'))
    b=subprocess.check_output(['git','-C',str(official),'show',commit+':'+rel])
    assert b==p.read_bytes()
    report['official_source_parity'][rel]=sha(p)
targets={}
for r in rows:
    a=torch.load(r['path'],map_location='cpu',weights_only=False)
    y=np.asarray(a['target']);obs=np.asarray(a['observed'])
    assert obs.dtype==bool and np.array_equal(y,y.T) and np.array_equal(obs,obs.T)
    assert set(np.unique(y)).issubset({0,1})
    mask=(np.arange(len(y))[:,None]<np.arange(len(y))[None,:]) & ~obs
    assert not (mask & obs).any() and not np.diag(mask).any()
    targets[r['key']]=(y,obs,mask)
    off=~np.eye(len(y),dtype=bool)
    report['graphs'].append(dict(key=r['key'],nodes=len(y),pairs=int(mask.sum()),positives=int(y[mask].sum()),constant_missing_ap=ap(y[mask],np.zeros(mask.sum())),copy_observed_full_ap=ap(y[off],obs[off].astype(float))))
for model,root in roots.items():
    m=json.loads((root/'metrics.json').read_text());gg=[];ys=[];ss=[];files={}
    for key,(y,obs,mask) in targets.items():
        p=root/'val'/(key+'.npz')
        with np.load(p) as v:
            assert np.array_equal(v['target'],y) and np.array_equal(v['observed'],obs)
            score=v['score'];assert score.shape==y.shape and np.isfinite(score).all()
            assert np.max(np.abs(score-score.T))<1e-5
            yy=y[mask];s=score[mask];off=~np.eye(len(y),dtype=bool)
            gg.append(dict(key=key,missing_ap=ap(yy,s),full_ap=ap(y[off],score[off]),unique_scores=int(len(np.unique(s)))))
            ys.append(yy);ss.append(s);files[str(p)]=sha(p)
    macro=float(np.mean([g['missing_ap'] for g in gg]));micro=ap(np.concatenate(ys),np.concatenate(ss))
    full=float(np.mean([g['full_ap'] for g in gg]))
    assert abs(macro-m['val']['missing_edge_macro_ap'])<1e-12
    assert abs(micro-m['val']['missing_edge_micro_ap'])<1e-12
    assert abs(full-m['val']['full_matrix_macro_ap'])<1e-12
    assert m['epochs']==80 and m['optimizer_steps']==12480
    assert sha(root/'final.pt')==m['final_sha256']
    report['methods'][model]=dict(macro_ap=macro,micro_ap=micro,full_adjacency_macro_ap=full,epochs=80,updates=12480,checkpoint_sha256=m['final_sha256'],metrics_sha256=sha(root/'metrics.json'),prediction_hashes=files,graphs=gg)
# R28 targets and raw observed support must match too; score readout is task-specific.
for key,(y,obs,mask) in targets.items():
    with np.load(r28/('prediction_'+key+'.npz')) as v:
        assert np.array_equal(v['target'],y) and np.array_equal(v['observed'],obs)
adapt=json.loads((roots['tlpnet']/'metrics.json').read_text())
nat=json.loads((native/'metrics.json').read_text())
assert adapt['identity']['initial_state_sha256']==nat['identity']['initial_state_sha256']
h1=json.loads((roots['tlpnet']/'history.json').read_text());h2=json.loads((native/'history.json').read_text())
assert len(h1)==len(h2)==80
assert all(a['sample_order_sha256']==b['sample_order_sha256'] for a,b in zip(h1,h2))
report['same_initialization_and_80_epoch_orders']=True
report['counts']=dict(graphs=62,pairs=sum(g['pairs'] for g in report['graphs']),positives=sum(g['positives'] for g in report['graphs']),zero_positive_graphs=sum(g['positives']==0 for g in report['graphs']),constant_macro_ap=float(np.mean([g['constant_missing_ap'] for g in report['graphs']])),copy_observed_full_macro_ap=float(np.mean([g['copy_observed_full_ap'] for g in report['graphs']])))
report['r28_target_observed_parity_graphs']=62
report['passed']=True
print(json.dumps(report,indent=2))

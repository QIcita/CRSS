"""Compute both TLP AP scopes from existing development predictions; never train."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
base=Path('/root/megrez-tmp/CPSF_NREDD_BASELINES_20261014')
rows=[r for r in json.loads((base/'data/tlp/manifest.json').read_text())['records'] if r['split']=='val']
assert len(rows)==62
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
configs={k:dict(root=base/'results'/(k+'_tlp'),prefix='val/') for k in ['sgformer','polynormer','tlpnet']}
configs['tlpnet_native_objective']=dict(root=Path('/root/megrez-tmp/CPSF_TLP_NATIVE_OBJECTIVE_20261015'),prefix='val/')
configs['r9']=dict(root=Path('/root/megrez-tmp/CPSF_R9_TLP_EVAL_REPAIR_20261018_A1'),prefix='prediction_',checkpoint=Path('/root/megrez-tmp/CPSF_R9_TASK_EXPANSION_20261017_A2/results/patch_rms_010_tlp/final.pt'))
configs['r28']=dict(root=Path('/root/megrez-tmp/CPSF_R28_TASK_EXPANSION_20261020_A2/results/r28_lpa_attention_tlp'),prefix='prediction_')
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),split='development validation',seed=42,graphs=62,test_predictions_accessed=False,test_metrics_used=False,training_launched=False,observed_scores_overwritten=False,missing_scope='i<j AND NOT raw_observed',full_scope='all off-diagonal matrix entries, including observed edges; raw saved model probabilities, no copying observed edges',macro='unweighted arithmetic mean of per-graph AP',methods={})
source={}
for r in rows:
 a=torch.load(r['path'],map_location='cpu',weights_only=False);source[r['key']]=(np.asarray(a['target']),np.asarray(a['observed']))
for name,c in configs.items():
 root=c['root'];m=json.loads((root/'metrics.json').read_text());blocks=[];files={}
 for key,(y,obs) in source.items():
  p=root/(c['prefix']+key+'.npz')
  with np.load(p) as v:
   assert np.array_equal(v['target'],y) and np.array_equal(v['observed'],obs)
   score=v['score'];assert np.isfinite(score).all() and score.shape==y.shape
   mask=np.triu(~obs,1);off=~np.eye(len(y),dtype=bool)
   assert not np.diag(mask).any() and not (mask&obs).any()
   blocks.append(dict(key=key,missing_edge_ap=float(average_precision_score(y[mask],score[mask])),full_adjacency_ap=float(average_precision_score(y[off],score[off])),missing_pairs=int(mask.sum()),missing_positives=int(y[mask].sum()),full_entries=int(off.sum()),full_positives=int(y[off].sum())))
  files[str(p)]=sha(p)
 missing=float(np.mean([b['missing_edge_ap'] for b in blocks]));full=float(np.mean([b['full_adjacency_ap'] for b in blocks]))
 assert abs(missing-m['val']['missing_edge_macro_ap'])<1e-12
 if 'full_matrix_macro_ap' in m['val']:assert abs(full-m['val']['full_matrix_macro_ap'])<1e-12
 ck=c.get('checkpoint',root/'final.pt');assert sha(ck)==m['final_sha256']
 report['methods'][name]=dict(missing_edge_macro_ap=missing,full_adjacency_macro_ap=full,source_metrics=str(root/'metrics.json'),metrics_sha256=sha(root/'metrics.json'),checkpoint=str(ck),checkpoint_sha256=m['final_sha256'],graphs=blocks,prediction_hashes=files)
report['passed']=True
print(json.dumps(report,indent=2))

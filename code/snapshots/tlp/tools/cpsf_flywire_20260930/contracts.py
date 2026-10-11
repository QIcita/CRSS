"""Real geometry integration plus identity/split/area/metric contracts on CPU."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .task import read_views,CellModel
from .evaluation import retrieve,classify
from .build import save,sha

def run(root):
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(1)
 rows=[json.loads(line) for line in (root/'metadata/manifest_provisional.jsonl').read_text().splitlines()]
 assert len(rows)==len({r['root_id'] for r in rows})==8122
 groups={s:{r['root_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
 assert all(groups[a].isdisjoint(groups[b]) for a,b in [('train','val'),('train','test'),('val','test')])
 for typ in {r['cell_type'] for r in rows}:
  c=Counter(r['split'] for r in rows if r['cell_type']==typ);assert c['train']>=3 and c['val']>=1 and c['test']>=1
 report=json.loads((root/'pilot_build.json').read_text());assert report['processed']==6 and all(r['status']=='ok' for r in report['records'])
 checks=[]
 for row in report['records']:
  path=root/'samples'/(row['root_id']+'.npz');assert sha(path)==row['sample_sha256']
  with np.load(path) as x:
   assert 2<=int(x['view_count'])<=33
   assert abs(row['tile_area_sum_um2']-row['total_area_um2'])/row['total_area_um2']<1e-9
   for v in row['views']:
    k=f'v{v["index"]:02d}_';assert np.isclose(x[k+'surface_weight_um2'].sum(),v['area_um2'],rtol=2e-6)
  # Full variable-size cell forward; small shared query-view backward below bounds smoke memory.
  gs=read_views(path,'measure');m=CellModel(6).eval()
  with torch.no_grad():feature=m.embedding(gs);logits=m(gs)
  assert feature.shape==(128,) and logits.shape==(1,6) and torch.isfinite(logits).all()
  ps=read_views(path,'point')
  for g,p in zip(gs,ps):
   assert p['n']==0 and len(p['e'])==0 and len(p['sf'])==0
   for key in ('p','pf','c','mass','w'):torch.testing.assert_close(g[key],p[key],atol=0,rtol=0)
  checks.append(dict(root_id=row['root_id'],views=len(gs),unit='um',full_cell_forward_finite=True,point_has_no_graph=True,matching_surface_tokenization=True,coverage=row['coverage']))
  del gs,ps,m
 first=root/'samples'/(report['records'][0]['root_id']+'.npz');grads={}
 for arm in ('measure','count','point','skeleton','geometry'):
  torch.manual_seed(42);m=CellModel(148,arm).train();gs=read_views(first,arm)[:1];out=m(gs);loss=torch.nn.functional.cross_entropy(out,torch.tensor([0]));loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  grads[arm]=float(sum(p.grad.square().sum() for p in m.parameters() if p.grad is not None).sqrt());assert grads[arm]>0
  if arm=='point':assert all(p.grad is None for p in m.encoder.graph_stem.parameters())
  del m,gs
 q=np.eye(2);g=np.repeat(q,3,axis=0);r=retrieve(q,g,['a','b'],['a']*3+['b']*3,['qa','qb'],['g'+str(i) for i in range(6)])
 assert r['query_macro_mAP']==r['cell_type_macro_mAP']==r['MRR']==1 and r['recall']['1']==1/3 and r['hit_rate']['1']==1
 try:retrieve(q,g,['a','x'],['a']*3+['b']*3,['qa','qx'],['g'+str(i) for i in range(6)]);raise RuntimeError('Zero-positive query silently accepted')
 except AssertionError:pass
 c=classify([0,0,1,1],[[2,0],[2,0],[0,2],[0,2]],2);assert c['macro_f1']==1
 save(root/'pilot_contract.json',dict(passed=True,scope='engineering checks only; no FlyWire task accuracy',seed=42,geometry_checks=checks,gradient_norms=grads,split_disjoint=True,minimum_train_gallery_positives=3,area_partition_conserved=True,retrieval_known_answer=True,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}));print('FLYWIRE PILOT CONTRACTS PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

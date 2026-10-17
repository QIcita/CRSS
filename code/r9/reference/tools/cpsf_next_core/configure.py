from .common import *
from .sampling import CONFIG

def main():
 assert not (ROOT/'protocol.json').exists()
 parent=BASE/'CPSF_EXTERNAL_BASELINES_PHASE2_20261010/data/manifest.json';split=BASE/'CPSF_FIXED_CORE_NREDD_20261006/split_manifest.json'
 source=json.loads(parent.read_text());s=json.loads(split.read_text());refs={str(parent):sha(parent),str(split):sha(split)}
 baseline=json.loads((ROOT/'baseline_reference.json').read_text());refs[str(ROOT/'baseline_reference.json')]=sha(ROOT/'baseline_reference.json')
 # Verify cited baseline artifacts before freezing a comparison gate.
 for r in baseline['records']:
  assert sha(r['evaluation_source'])==r['evaluation_source_sha256'],r['evaluation_source']
  refs[r['evaluation_source']]=r['evaluation_source_sha256']
 gates={}
 for task in ['h01_local','h01_global','flywire_coarse']:
  rr=[r for r in baseline['records'] if r['task']==task]
  best=(min if task=='h01_global' else max)(rr,key=lambda r:r['value'])
  gates[task]=dict(metric=best['primary_metric'],direction='min' if task=='h01_global' else 'max',baseline=best['value'],model=best['model'],source=best['evaluation_source'])
 # Add GraPHFormer adapted (extra skeleton radius) conservatively to strongest-baseline gate.
 for task in ['h01_local','h01_global','flywire_coarse']:
  path=BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013/results'/task/'metrics.json';d=json.loads(path.read_text());refs[str(path)]=sha(path);value=d['val'][gates[task]['metric']]
  if (value<gates[task]['baseline'] if task=='h01_global' else value>gates[task]['baseline']):gates[task].update(baseline=value,model='GraPHFormer-adapted',source=str(path))
 # Use same-split validation for adaptive development, never test as optimizer feedback.
 path=BASE/'CPSF_NREDD_BASELINES_20261014/results/ted_tednet/metrics.json'
 if not path.exists():
  candidates=[]
  for x in (BASE/'CPSF_NREDD_BASELINES_20261014/results').glob('*/metrics.json'):
   d=json.loads(x.read_text())
   if 'f1' in d.get('val',{}):candidates.append((d['val']['f1'],x,d))
  value,path,d=max(candidates,key=lambda x:x[0])
 else:d=json.loads(path.read_text());value=d['val']['f1']
 refs[str(path)]=sha(path);gates['ted']=dict(metric='f1',direction='max',baseline=value,model=path.parent.name,source=str(path),test_policy='test only after final architecture freeze, not hourly tuning')
 # Keep old complete-model performance as additional no-regression target for H01.
 for task in ['h01_local','h01_global']:
  path=BASE/'CPSF_V26_COMPLETE_20261003/results'/f'{task}_fixed_residual'/'metrics.json';d=json.loads(path.read_text());refs[str(path)]=sha(path)
  gates[task]['retained_full']=d['val'][gates[task]['metric']]
  gates[task]['progression_threshold']=(min if task=='h01_global' else max)(gates[task]['baseline'],gates[task]['retained_full'])
 gates['ted']['progression_threshold']=gates['ted']['baseline']
 save(ROOT/'input_manifest.json',dict(h01={t:[r for r in source['records'] if r['task']==t] for t in ['h01_local','h01_global','flywire_coarse']},ted_blocks=[r for r in s['records']['ted'] if r['split'] in ['train','val']]))
 refs[str(ROOT/'input_manifest.json')]=sha(ROOT/'input_manifest.json')
 p=dict(version=ROOT.name,seed=42,stage='first three tasks',tasks={'h01_local':dict(epochs=25,batch=8,micro=4,lr=.0003,weight_decay=.01,lr_min=.00003,train=716,val=214,updates=2250),
  'h01_global':dict(epochs=50,batch=4,micro=1,lr=.0003,weight_decay=.01,lr_min=.00003,train=80,val=24,updates=1000),
  'ted':dict(epochs=30,batch=64,micro=16,lr=.0005,weight_decay=.0005,train=23967,val=10762,updates=11250)},
  sampling=CONFIG,network=dict(width=96,layers=3,heads=4,rope_base=10,context_tokens=48,residual=.1,pretrained=False),
  progression_gates=gates,selection='fixed final epoch; same original development splits; no automatic test feedback',
  benchmark_scope='strongest verified reproduced baseline under project protocol; not a universal SOTA claim',
  baseline_scope='same source observation support, labels, split, epoch and update budgets; algorithm/tokenization changes are method contributions, not identical internal tensors or FLOPs',
  historical_exposure='development sets and historical tests previously used; no claim of pristine blind evaluation',
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},parent_sha256=refs)
 save(ROOT/'protocol.json',p);verify();print(json.dumps(gates,indent=2))
if __name__=='__main__':main()

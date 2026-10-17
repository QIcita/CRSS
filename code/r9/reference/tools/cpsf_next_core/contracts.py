import gc,time,copy
from .common import *
from .sampling import coarsen,CONFIG,baseline_anchors
from .data import pack
from .model import Model,rotate
from .train import Cache
from scipy.spatial import cKDTree

def geometry_checks():
 rng=np.random.RandomState(42);p=rng.normal(size=(511,3));w=rng.uniform(.1,2,len(p));cfg={k:v for k,v in CONFIG.items() if k in ['min_tokens','max_tokens','relative_rmse','max_relative_radius','grain_radius_factor','lloyd_steps']}
 a=coarsen(p,w,**cfg);b=coarsen(np.repeat(p,2,axis=0),np.repeat(w/2,2),**cfg);c=coarsen(p*3+[20,30,40],w,**cfg)
 np.testing.assert_allclose(a['pos'],b['pos'],atol=1e-10);np.testing.assert_allclose(a['mass'],b['mass'],atol=1e-10)
 np.testing.assert_allclose(c['pos'],a['pos']*3+[20,30,40],atol=1e-8);np.testing.assert_allclose(c['scale'],a['scale']*3,atol=1e-8)
 assert abs(a['mass'].sum()-w.sum())<1e-8
 q=torch.randn(9,4,24);k=torch.randn_like(q);x=torch.randn(9,3);y=torch.randn(9,3)
 left=(rotate(q,x)*rotate(k,y)).sum(-1);right=(q*rotate(k,y-x)).sum(-1);assert (left-right).abs().max()<1e-5
 return dict(mass_conserved=True,duplicate_split_mass_invariant=True,translation_and_uniform_scale_covariant_sampling=True,relative_rope_identity_maxdiff=float((left-right).abs().max()))

def sampling_benchmark(task,record):
 # All methods receive the exact same observed support and weight; same K.
 if task=='ted':
  source=record['source'];block=next(r for r in json.loads((ROOT/'input_manifest.json').read_text())['ted_blocks'] if r['key']==source['block'])
  from tools.cpsf_fixed_nredd.data import ted_events
  ev=next(e for e in ted_events(block) if e['key']==record['key']);points=ev['p'];weights=np.ones(len(points))
 else:
  with np.load(record['source']['point_path']) as z:points=z['coord'].astype(float);weights=z['mass'].astype(float)
 t=time.perf_counter();cfg={k:v for k,v in CONFIG.items() if k in ['min_tokens','max_tokens','relative_rmse','max_relative_radius','grain_radius_factor','lloyd_steps']};a=coarsen(points,weights,**cfg);seconds=time.perf_counter()-t;k=len(a['pos']);p=a['points'];w=a['weights'];report={'adaptive':dict(k=k,seconds=seconds,quantization_mse=a['stats']['distortion'],max_radius=float(a['radius'].max()))}
 for method in ['random','uniform','fps']:
  t=time.perf_counter();anchors=baseline_anchors(p,k,method);distance,owner=cKDTree(anchors).query(p);mass=np.bincount(owner,weights=w,minlength=k);centers=np.zeros((k,3));np.add.at(centers,owner,p*w[:,None]);centers/=np.maximum(mass,1e-12)[:,None]
  # Give comparators identical full-point centroid/mass reduction, not subset-only treatment.
  residual=np.linalg.norm(p-centers[owner],axis=1)
  report[method]=dict(k=k,seconds=time.perf_counter()-t,quantization_mse=float(np.sum(w*residual**2)/w.sum()),max_radius=float(residual.max()),empty_cells=int((mass==0).sum()))
 return dict(task=task,key=record['key'],scope='same observed support and K; all methods get full-point mass/centroid aggregation; engineering not accuracy',methods=report)

def main():
 setup();verify();result=dict(passed=False,protocol_sha256=sha(ROOT/'protocol.json'),geometry=geometry_checks(),tasks={},sampling_benchmark=[])
 for task in ['h01_local','h01_global','ted']:
  rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
  order=sorted(tr,key=lambda r:(r['skeleton_nodes']+r['tokens'],r['key']));chosen=[order[0],order[len(order)//2],order[-1]];rowsout=[]
  setup();model=Model(task).cuda();initial=state_sha(model)
  for record in chosen:
   sample=load_item(record);torch.cuda.reset_peak_memory_stats();model.eval();x=pack([sample])
   with torch.no_grad():
    first=model(x);batch=model(pack([sample,sample]));again=model(x)
   assert torch.equal(first,again);assert (first-batch[:1]).abs().max()<2e-5
   translated={k:v.clone() for k,v in sample.items()}
   for field in ['spos','ppos','cpos','query']:translated[field]+=torch.tensor([10.,20.,30.])
   with torch.no_grad():translated_output=model(pack([translated]))
   assert (first-translated_output).abs().max()<2e-4,(task,'translation',float((first-translated_output).abs().max()))
   model.train();model.zero_grad(set_to_none=True);z=model(x);z.square().mean().backward();active=[n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any())]
   assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
   for prefix in ['graph.','point.','patch.']+[f'blocks.{i}.{m}.' for i in range(3) for m in ['graph','point','cross','context']]:assert any(n.startswith(prefix) for n in active),prefix
   torch.optim.AdamW(model.parameters(),lr=1e-4).step();assert all(torch.isfinite(v).all() for v in model.parameters())
   rowsout.append(dict(key=record['key'],nodes=record['skeleton_nodes'],tokens=record['tokens'],peak_bytes=torch.cuda.max_memory_allocated(),batch_parity_maxdiff=float((first-batch[:1]).abs().max()),translation_maxdiff=float((first-translated_output).abs().max()),active_gradient_parameters=active))
   del x,z,batch,first,again;gc.collect();torch.cuda.empty_cache()
  result['tasks'][task]=dict(initial_sha256=initial,parameters=sum(v.numel() for v in model.parameters()),cases=rowsout)
  result['sampling_benchmark'].append(sampling_benchmark(task,chosen[1]));del model;gc.collect();torch.cuda.empty_cache()
 result['passed']=True;save(ROOT/'contracts.json',result);print(json.dumps({task:max(r['peak_bytes'] for r in d['cases']) for task,d in result['tasks'].items()}),flush=True)
if __name__=='__main__':main()

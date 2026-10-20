"""Real train samples, max support, strict observed-source and isolation checks."""
import gc,copy,time
from .common import *
from .data import pack
from .model import Model

def main():
 setup();protocol=verify();result=dict(passed=False,protocol_sha256=sha(ROOT/'protocol.json'),tasks={},test_used=False)
 for task in protocol['tasks']:
  rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];cfg=protocol['tasks'][task]
  assert (len(tr),len(va))==(cfg['train'],cfg['val']);assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
  order=sorted(tr,key=lambda r:r['incidences']+r['nodes']+r['points']);chosen=[order[0],order[len(order)//2],order[-1]]
  if task=='ted':chosen.append(next(r for r in tr if r['target']==1))
  setup();model=Model(task).cuda();active=set();results=[]
  for record in chosen:
   print('acceptance',task,record['key'],record['incidences'],flush=True)
   sample=load_item(record);assert 'target' not in sample and sample['pf'].shape[1]==9;other=load_item(order[0]);torch.cuda.reset_peak_memory_stats();model.eval();g=pack([sample])
   with torch.no_grad():
    first=model(g);mixed=model(pack([sample,other]));again=model(g)
   assert torch.equal(first,again);assert (first-mixed[:1]).abs().max()<5e-5,(task,'batch',float((first-mixed[:1]).abs().max()))
   moved={k:v.clone() for k,v in sample.items()}
   for key in ['s','pos','patch_pos','query']:moved[key]+=torch.tensor([10.,20.,30.])
   with torch.no_grad():translated=model(pack([moved]))
   assert (first-translated).abs().max()<5e-4,(task,'translation',float((first-translated).abs().max()))
   # Node identifiers do not encode labels: permutation plus incidence remapping.
   perm=torch.arange(len(sample['s'])-1,-1,-1);inv=torch.empty_like(perm);inv[perm]=torch.arange(len(perm));reindexed={k:v.clone() for k,v in sample.items()};reindexed['s']=sample['s'][perm];reindexed['sf']=sample['sf'][perm];reindexed['sn']=inv[sample['sn']]
   with torch.no_grad():renumbered=model(pack([reindexed]))
   assert (first-renumbered).abs().max()<5e-5
   model.train();model.zero_grad(set_to_none=True);t=time.perf_counter();z=model(g);z.square().sum().backward();torch.cuda.synchronize();seconds=time.perf_counter()-t
   assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters());active.update(n for n,v in model.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
   torch.optim.AdamW(model.parameters(),lr=1e-4).step();assert all(torch.isfinite(v).all() for v in model.parameters())
   results.append(dict(key=record['key'],nodes=record['nodes'],regions=record['regions'],points=record['points'],incidences=record['incidences'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),forward_backward_seconds=seconds,batch_maxdiff=float((first-mixed[:1]).abs().max()),translation_maxdiff=float((first-translated).abs().max()),node_renumber_maxdiff=float((first-renumbered).abs().max())))
   del g,z,first,mixed,again,translated,renumbered;gc.collect();torch.cuda.empty_cache()
  for prefix in ['position.','fusion.graph.','fusion.point.','fusion.path.','fusion.patch.','fusion.geometry.','fusion.interaction.','fusion.scale.']+[f'layers.{i}.{n}.' for i in range(2) for n in ['qkv','edge','geometry','cqkv']]:assert any(n.startswith(prefix) for n in active),(task,prefix)
  # Match maximum per-task microbatch geometry, without reading labels.
  largest=load_item(order[-1]);model.train();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
  z=model(pack([largest]*cfg['micro']));z.square().sum().backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
  micro_peak=torch.cuda.max_memory_allocated();del z;model.zero_grad(set_to_none=True)
  result['tasks'][task]=dict(max_microbatch_peak_gpu_bytes=micro_peak,parameters=sum(v.numel() for v in model.parameters()),cases=results,active_gradient_parameters=sorted(active));del model;gc.collect();torch.cuda.empty_cache()
 result['passed']=True;save(ROOT/'contracts.json',result);print(json.dumps({t:max(x['peak_gpu_bytes'] for x in d['cases']) for t,d in result['tasks'].items()}),flush=True)
if __name__=='__main__':main()

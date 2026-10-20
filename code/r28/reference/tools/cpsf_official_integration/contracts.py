import gc,copy,inspect,time
from .common import *
from .data import move
from .model import Model
from .train import calc
from tools.cpsf_fixed_core.model import Model as OldH01
from tools.cpsf_fixed_nredd.model import Model as OldTED

def old_model(task):return OldTED(task,'full') if task=='ted' else OldH01(task,'full')
def main():
 setup();p=verify();result=dict(passed=False,protocol_sha256=sha(ROOT/'protocol.json'),reference={},cases={},test_used=False)
 # Reference checkpoint replay, no training or model selection.
 for task in ['h01_local','h01_global']:
  ref=p['retained_references'][task];rr=[r for r in rows(task) if r['split']=='val'];m=Model(task,'none').cuda().eval();ck=torch.load(Path(ref['path'])/'final.pt',map_location='cpu',weights_only=False);m.load_state_dict({k:ck['model'][k] for k in m.state_dict()});zz=[]
  with torch.no_grad():
   for r in rr:zz.append(m(move(load_item(r))).cpu().numpy())
  original=np.load(Path(ref['path'])/'val_predictions.npz');assert original['key'].tolist()==[r['key'] for r in rr];np.testing.assert_allclose(zz,original['prediction_or_logits'],atol=2e-5,rtol=2e-5)
  metric=calc([r['target'] for r in rr],zz,task);key='mae' if task=='h01_global' else 'macro_f1';assert abs(metric[key]-ref['val'][key])<1e-9
  result['reference'][task]=dict(metrics=metric,prediction_maxdiff=float(abs(np.array(zz)-original['prediction_or_logits']).max()));del m;gc.collect();torch.cuda.empty_cache();print('reference',task,metric,flush=True)
 for kind in ['utonia','geotransformer']:
  for task in p['tasks']:
   rr=rows(task);tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val'];assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
   order=sorted(tr,key=lambda r:r['incidences']+r['nodes']+r['points']);chosen=[order[0],order[len(order)//2],order[-1]];setup();base=old_model(task).cuda();setup();m=Model(task,kind).cuda()
   assert all(torch.equal(v,base.state_dict()[k]) for k,v in m.state_dict().items() if not k.startswith('native_'));active=set();cases=[]
   for r in chosen:
    print('case',kind,task,r['key'],flush=True);g=move(load_item(r));base.load_state_dict({k:m.state_dict()[k] for k in base.state_dict()});m.eval();base.eval();m.integration_enabled=False
    with torch.no_grad():assert torch.equal(m(g),base(g))
    m.integration_enabled=True;torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():a=m(g);b=m(g);assert torch.allclose(a,b,rtol=1e-5,atol=1e-5)
    other=move(load_item(order[0]))
    with torch.no_grad():m(other);after=m(g);assert torch.allclose(a,after,rtol=1e-5,atol=1e-5)
    moved=dict(g)
    for key in ['s','pos','query','query_xyz']:
     if key in moved:moved[key]=moved[key]+moved[key].new_tensor([1.,-2.,3.])
    with torch.no_grad():shifted=m(moved);assert torch.allclose(a,shifted,rtol=3e-4,atol=3e-4),(kind,task,'translation',a,shifted)
    m.train();m.zero_grad(set_to_none=True);start=time.perf_counter();z=m(g);z.square().sum().backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());active.update(n for n,v in m.named_parameters() if v.grad is not None and bool((v.grad!=0).any()))
    torch.optim.AdamW(m.parameters(),lr=1e-4).step();assert all(torch.isfinite(v).all() for v in m.parameters())
    # Checkpoint vs direct original Utonia block execution on real median geometry.
    parity=None
    if kind=='utonia' and r==chosen[1]:
     m.train();m.zero_grad(set_to_none=True);z=m(g);z.square().sum().backward();first={n:v.grad.detach().clone() for n,v in m.named_parameters() if v.grad is not None};value=z.detach().clone()
     for block in m.native_point.encoder.modules():
      if hasattr(block,'enabled'):block.enabled=False
     m.zero_grad(set_to_none=True);z=m(g);z.square().sum().backward();maxabs=max(float((v.grad-first[n]).abs().max()) for n,v in m.named_parameters() if n in first);scale=max(float(v.abs().max()) for v in first.values());assert torch.allclose(value,z,rtol=2e-4,atol=2e-4);assert maxabs/max(scale,1e-12)<2e-3
     parity=dict(output_maxdiff=float((value-z).abs().max()),gradient_max_relative_to_global_max=maxabs/max(scale,1e-12))
     for block in m.native_point.encoder.modules():
      if hasattr(block,'enabled'):block.enabled=True
    cases.append(dict(key=r['key'],forward_backward_seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_parity=parity,groups=len(g['geo_groups']),point_voxels=len(g['ufeat'])));del g,other,moved,z;gc.collect();torch.cuda.empty_cache()
   for prefix in ['graph.','point.','position.','scale.','processor.','joint.']+(['native_point.encoder.embedding.','native_point.encoder.enc.enc0.','native_point.encoder.enc.enc1.','native_point.encoder.enc.enc2.','native_point.proj.'] if kind=='utonia' else ['native_joint.transformer.embedding.','native_joint.transformer.transformer.']):assert any(n.startswith(prefix) for n in active),(kind,task,prefix)
   sources={}
   for n,module in m.named_modules():
    try:path=inspect.getfile(type(module))
    except TypeError:continue
    if '/vendor/' in path:sources[n]=dict(class_name=type(module).__name__,path=path)
   assert sources
   result['cases'][kind+'_'+task]=dict(cases=cases,parameters=sum(v.numel() for v in m.parameters()),active_gradients=sorted(active),official_classes=sources);del m,base;gc.collect();torch.cuda.empty_cache()
 result['passed']=True;save(ROOT/'contracts.json',result);print('ALL PASSED',flush=True)
if __name__=='__main__':main()

"""Reuse validated observed arclength supports, repair singleton point geometry.
Only train/dev; targets stay in manifests, never in model input dictionaries.
"""
import time,shutil
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from .common import *
from .data import raw_graph
R2=BASE/'CPSF_NEXT_CORE_20261015_R2'

def write(record,source_s,points,weights):
 assert sha(record['path'])==record['sha256'];g=raw_graph(record['path']);s=np.asarray(source_s,float);p=np.asarray(points,float);w=np.asarray(weights,float)
 assert len(s)==len(g['s']);offset=np.median(s-g['s'].numpy(),axis=0);np.testing.assert_allclose(s-offset,g['s'].numpy(),atol=.003,rtol=1e-6)
 p,inv=np.unique(p-offset,axis=0,return_inverse=True);w=np.bincount(inv,weights=w);owner=g['point_owner'].numpy();assert len(p)==len(owner)
 np.testing.assert_allclose(w,g['point_weight'].numpy(),atol=.003,rtol=2e-5)
 mass=np.bincount(owner,weights=w,minlength=g['patches']);center=np.zeros((len(mass),3));np.add.at(center,owner,p*w[:,None]);center/=mass[:,None]
 oldrel=(p-center[owner])/.5;err=float(abs(oldrel-g['pf'].numpy()[:,:3]).max());assert err<.003,(record['key'],'source-point registration mismatch',err)
 _,ix=cKDTree(p).query(p,k=min(16,len(p)));ix=np.asarray(ix).reshape(len(p),-1);near=p[ix];nw=w[ix];c=(near*nw[:,:,None]).sum(1)/nw.sum(1)[:,None];h=np.sqrt((((near-c[:,None])**2).sum(-1)*nw).sum(1)/nw.sum(1)).clip(1e-6)
 rel=(p-c)/h[:,None];feature=np.column_stack((g['pf'].numpy(),np.clip(rel,-12,12),np.linalg.norm(rel,axis=1),np.log1p(h))).astype(np.float16)
 assert feature.shape==(len(p),9) and np.isfinite(feature).all()
 path=ROOT/'data'/record['task']/(record['key'].replace(':','_')+'.features.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
 np.savez_compressed(path,pf=feature,patch_pos=center.astype(np.float32));assert shutil.disk_usage(ROOT).free>4*2**30
 return dict(record,feature_path=str(path),feature_sha256=sha(path),nodes=g['raw_nodes'],regions=g['nregions'],patches=g['patches'],points=len(p),incidences=len(g['pr']),source_registration_max_error=err,zero_old_fraction=float((g['pf'].numpy()==0).all(1).mean()),zero_new_fraction=float((feature==0).all(1).mean()),all_observed_nodes_preserved=True)

def h01(pair):
 r,source=pair;torch.set_num_threads(1)
 for key in ['graph','point']:assert sha(source[key+'_path'])==source[key+'_sha256']
 if r['task']=='h01_global':
  # Reproduce frozen thumbnail+tile draws before float32 export. Sorting the
  # rounded export can swap near-equal points and their unequal masses.
  assert sha(source['source_path'])==source['source_sha256']
  with np.load(source['source_path']) as z:
   points=z['surface_xyz_um'];skeleton=z['skeleton_xyz_um'];rng=np.random.default_rng(42);ids=np.arange(len(points));chosen=[np.sort(rng.choice(ids,min(len(ids),4096),replace=False))];weights=[np.full(len(chosen[0]),.5*len(points)/len(chosen[0]))];chunks=[ids];target=min(32,max(1,int(np.ceil(len(points)/16384))))
   while len(chunks)<target:
    k=max(range(len(chunks)),key=lambda i:len(chunks[i]));ids=chunks.pop(k)
    if len(ids)<2:chunks.insert(k,ids);break
    axis=np.ptp(points[ids],axis=0).argmax();ids=ids[np.argsort(points[ids,axis],kind='stable')];mid=len(ids)//2;chunks[k:k]=[ids[:mid],ids[mid:]]
   for ids in chunks:
    take=np.sort(rng.choice(ids,min(len(ids),2048),replace=False));chosen.append(take);weights.append(np.full(len(take),.5*len(ids)/len(take)))
   return [write(r,skeleton,points[np.concatenate(chosen)],np.concatenate(weights))]
 with np.load(source['graph_path']) as a,np.load(source['point_path']) as z:return [write(r,a['coord'],z['coord'],z['mass'])]

def ted(pair):
 block,records=pair;torch.set_num_threads(1)
 from tools.cpsf_fixed_nredd.data import ted_events
 bykey={r['key']:r for r in records};out=[]
 for e in ted_events(block):
  r=bykey[e['key']];assert r['target']==int(e['target'].argmax());out.append(write(r,e['s'],e['p'],np.ones(len(e['p']))))
 assert len(out)==len(records);return out

def main():
 verify();start=time.time();m=json.loads((ROOT/'input_manifest.json').read_text());summary={}
 for task in ['h01_local','h01_global','ted']:
  records=m['records'][task]
  if task=='ted':
   groups={}
   for r in records:groups.setdefault(r['source']['block'],[]).append(r)
   jobs=[(b,groups[b['key']]) for b in m['ted_blocks']];worker=ted
  else:jobs=[(r,r['source']) for r in records];worker=h01
  done=[]
  with ProcessPoolExecutor(max_workers=8) as pool:
   for i,x in enumerate(pool.map(worker,jobs,chunksize=1)):
    done+=x
    if i%20==0 or i+1==len(jobs):save(ROOT/'prepare_status.json',dict(task=task,items=i+1,total=len(jobs),samples=len(done),seconds=time.time()-start));print(task,i+1,len(jobs),len(done),flush=True)
  assert [r['key'] for r in done]==[r['key'] for r in records]
  assert {r['cell'] for r in done if r['split']=='train'}.isdisjoint(r['cell'] for r in done if r['split']=='val')
  save(ROOT/'data'/task/'manifest.json',dict(records=done,complete=True,protocol_sha256=sha(ROOT/'protocol.json'),labels_in_model_inputs=False))
  summary[task]=dict(samples=len(done),max_nodes=max(r['nodes'] for r in done),max_regions=max(r['regions'] for r in done),max_incidences=max(r['incidences'] for r in done),mean_zero_old=float(np.mean([r['zero_old_fraction'] for r in done])),mean_zero_new=float(np.mean([r['zero_new_fraction'] for r in done])),registration_max=max(r['source_registration_max_error'] for r in done))
 save(ROOT/'preparation.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),tasks=summary,seconds=time.time()-start,test_used=False))
if __name__=='__main__':main()

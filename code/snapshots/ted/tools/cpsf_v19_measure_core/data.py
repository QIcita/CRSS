"""Frozen cohorts; new geometry caches never reuse V18 voxelization."""
import argparse,time
from .io import *
from .geometry import prepare,Config,TED_CONFIG

def build_h01(out):
 assert not out.exists();out.mkdir(parents=True);(out/'local').mkdir()
 rows=json.loads((V9/'manifest.json').read_text())['records'];done=[];start=time.time()
 for cell in sorted({r['cell'] for r in rows}):
  source=SOURCE/'inputs'/(cell+'.npz');source_sha=sha(source)
  with np.load(source) as x:full={k:x[k] for k in ('skeleton_xyz_um','edge_index','surface_xyz_um','point_to_edge')}
  for r in (r for r in rows if r['cell']==cell):
   refpath=SOURCE/'local_refs'/(r['sample']+'.npz')
   with np.load(refpath) as ref:nodes=ref['node_ids'];points=ref['point_ids'];edges=ref['edge_ids'];q=int(ref['query_index'])
   edges=np.union1d(edges,np.unique(full['point_to_edge'][points]));nodes=np.r_[nodes,np.setdiff1d(np.unique(full['edge_index'][edges]),nodes)]
   lookup=np.full(len(full['skeleton_xyz_um']),-1,np.int64);lookup[nodes]=np.arange(len(nodes));s=full['skeleton_xyz_um'][nodes];e=lookup[full['edge_index'][edges]];p=full['surface_xyz_um'][points]
   assert len(p)==r['points'] and hashlib.sha256(points.tobytes()).hexdigest()==r['contract']['point_ids_sha256']
   # Recentre before float32 storage without a skeleton-dependent grid origin.
   center=s[q].copy();g=prepare(s-center,e,p-center)
   item=dict(geometry=g,query=q,target=int(r['target']),cell=cell,sample=r['sample'],split=r['split'])
   file=out/'local'/(r['sample']+'.pt');torch.save(item,file)
   done.append(dict(sample=r['sample'],cell=cell,split=r['split'],target=int(r['target']),points=len(p),sha256=sha(file),source_sha256=source_sha,reference_sha256=sha(refpath),wrong_fraction=float((g['sender']!=g['donor']).float().mean()) if len(g['sender']) else 0.))
  save(out/'progress.json',dict(samples=len(done),seconds=time.time()-start));print('CACHE',cell,len(done),flush=True)
 assert len(done)==930
 save(out/'manifest.json',dict(records=done,source_manifest_sha256=sha(V9/'manifest.json'),geometry=vars(Config()),measure='source empirical point count; no mesh area',seconds=time.time()-start))

def h01_sets(cache,smoke=False):
 rows=json.loads((cache/'manifest.json').read_text())['records'];sets={k:[] for k in ('train','val')}
 for r in rows:
  file=cache/'local'/(r['sample']+'.pt');assert sha(file)==r['sha256'];sets[r['split']].append(dict(r,path=file))
 assert [len(sets[k]) for k in sets]==[716,214]
 assert {r['cell'] for r in sets['train']}.isdisjoint(r['cell'] for r in sets['val'])
 if smoke:
  sets={k:[r for c in range(3) for r in [r for r in v if r['target']==c][:4]] for k,v in sets.items()}
 return sets

def ted_sets(smoke=False):
 report=json.loads((TED/'construction_report.json').read_text());sets={};audit=[]
 for split in ('train','test'):
  blocks=[]
  for row in report['splits'][split]['rows'][:1 if smoke else None]:
   file=TED/split/(Path(row['block']).stem+'.pt');assert sha(file)==row['sha256'];b=load(file)
   meta_file=NREDD/f'data/error_detect_data/{split}/event_cloud_cpsf_geometry_v2d_segment_shell'/(Path(row['block']).stem+'.npz')
   with np.load(meta_file) as x:
    meta=json.loads(x['metadata_json'].item());ids=x['subgraph_node_indices'];cached=x['skeleton_xyz']
   swc=Path(meta['source_swc']);assert sha(swc)==meta['source_swc_sha256'];xyz=np.loadtxt(swc,usecols=(2,3,4));events=[];maxerror=0.
   assert len(meta['event_statistics'])==len(b['events'])==len(ids)
   for i,event in enumerate(b['events'][:70 if smoke else None]):
    scale=float(meta['event_statistics'][i]['scale']);assert np.isfinite(scale) and scale>0
    s=event['xyz'].numpy();valid=ids[i][ids[i]>=0];assert len(s)==len(valid)
    np.testing.assert_allclose(s,cached[i,:len(s)],atol=0,rtol=0)
    raw=xyz[valid]-xyz[valid[0]];error=float(np.max(abs(s*scale-raw)));assert error<.002,(row['block'],i,error)
    maxerror=max(maxerror,error);p=np.concatenate([metric.numpy() for _,metric in event['patches']])
    events.append(dict(name=f'{Path(row["block"]).stem}:{i}',s=(s*scale).astype(np.float32),e=event['edges'].numpy(),p=(p*scale).astype(np.float32),target=b['labels'][i].clone(),scale=scale))
   blocks.append(dict(block=row['block'],events=events,source_sha256=row['sha256']))
   audit.append(dict(split=split,block=row['block'],events=len(events),maximum_swc_reconstruction_error=maxerror,metadata_sha256=sha(meta_file),swc_sha256=sha(swc)))
  sets[split]=blocks
 if not smoke:assert [sum(len(b['events']) for b in sets[k]) for k in sets]==[33519,9893]
 return sets,audit

def geometry(row,cache,task):
 if task=='h01':return cache.get(row['sample'],lambda:load(row['path'])['geometry'])
 return cache.get(row['name'],lambda:prepare(row['s'],row['e'],row['p'],TED_CONFIG,wrong=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();setup();build_h01(a.out)

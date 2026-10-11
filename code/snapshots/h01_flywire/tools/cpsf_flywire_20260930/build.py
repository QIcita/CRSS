"""Resumable CPU acquisition; per-root atomic success markers, bounded disk use."""
import argparse,struct,json,hashlib,time,os,shutil,fcntl,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from scipy.spatial import cKDTree
from cloudvolume import CloudVolume
from .sampling import build_views,draw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def npz(path,**data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**data)
 tmp.replace(path)
def get(url):
 response=requests.get(url,timeout=(15,120));response.raise_for_status();return response.content

def one(row,root):
 rid=row['root_id'];marker=root/'qa'/(rid+'.json');sample=root/'samples'/(rid+'.npz');raw=root/'raw'/rid;raw.mkdir(parents=True,exist_ok=True)
 if marker.exists():
  previous=json.loads(marker.read_text())
  if previous.get('status')=='ok':
   assert sample.is_file() and sha(sample)==previous['sample_sha256'];return previous
  # Failed records are immutable within a run. A new --retry-failed invocation archives the marker first.
  return previous
 assert shutil.disk_usage(root).free>50*2**30,'Disk headroom below 50GiB: pause without deleting data'
 skpath=raw/'skeleton.bin';meshpath=raw/'mesh.npz';skurl='https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/'+rid
 if not skpath.exists():
  b=get(skurl);tmp=skpath.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(skpath)
 b=skpath.read_bytes();assert len(b)>=8;n,ne=struct.unpack('<II',b[:8]);assert len(b)==8+16*n+8*ne and n>0 and ne>0
 s=np.frombuffer(b,dtype='<f4',count=3*n,offset=8).reshape(n,3).astype(np.float64)/1000
 e=np.frombuffer(b,dtype='<u4',count=2*ne,offset=8+12*n).reshape(ne,2).astype(np.int64)
 radius=np.frombuffer(b,dtype='<f4',count=n,offset=8+12*n+8*ne).astype(np.float64)/1000
 if not meshpath.exists():
  cv=CloudVolume('precomputed://https://storage.googleapis.com/flywire_v141_m783',progress=False,cache=False,parallel=1)
  mesh=cv.mesh.get(int(rid),lod=0)[int(rid)];npz(meshpath,vertices_nm=np.asarray(mesh.vertices,np.float32),faces=np.asarray(mesh.faces,np.int64))
 with np.load(meshpath) as mesh:v=mesh['vertices_nm'].astype(np.float64)/1000;f=mesh['faces'].astype(np.int64)
 assert len(v)>0 and len(f)>0 and np.isfinite(v).all() and np.isfinite(s).all() and np.isfinite(radius).all()
 assert e.min()>=0 and e.max()<n and f.min()>=0 and f.max()<len(v) and (e[:,0]!=e[:,1]).all()
 assert max(np.ptp(v,axis=0))<2000 and max(np.ptp(s,axis=0))<2000,'Implausible um span'
 # Point-derived centre only; soma coordinates, class, side, NT and connections never enter geometry.
 center=(v.min(0)+v.max(0))/2;s=s-center;v=v-center
 bounds_lo=v.min(0)-2;bounds_hi=v.max(0)+2;outside=float((~((s>=bounds_lo)&(s<=bounds_hi)).all(1)).mean())
 distance=cKDTree(v).query(s,k=1,workers=1)[0]
 flags=[]
 if outside>.05:flags.append('skeleton_outside_mesh_bbox_gt5pct')
 if float(np.quantile(distance,.95))>10:flags.append('skeleton_to_vertex_p95_gt10um')
 # Absolute vertex coordinates detect duplicated geometry across root IDs without conflating similar neurons.
 abs_v=v+center;order=np.lexsort((abs_v[:,2],abs_v[:,1],abs_v[:,0]));fingerprint=hashlib.sha256(abs_v[order].astype('<f8').tobytes()).hexdigest()
 arrays,view_audit=build_views(s,e,v,f)
 # Same-source, label-free surface coverage diagnostic. Reference is an independent seed42 substream.
 tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;valid=np.flatnonzero(area>1e-12)
 rng=np.random.default_rng(42);rng.bit_generator.advance(1000000)
 reference,_,_=draw(v,f,area,valid,4096,rng)
 cloud=np.concatenate([arrays[f'v{i:02d}_surface_xyz_um'] for i in range(int(arrays['view_count']))])
 old_distance=cKDTree(arrays['v00_surface_xyz_um']).query(reference,workers=1)[0];new_distance=cKDTree(cloud).query(reference,workers=1)[0]
 coverage=dict(reference_points=4096,unit='um',thumbnail_p95=float(np.quantile(old_distance,.95)),all_views_p95=float(np.quantile(new_distance,.95)),all_views_points=len(cloud),scope='Monte Carlo same-mesh surface coverage, not task accuracy')
 arrays['source_center_um']=center.astype(np.float64);npz(sample,**arrays)
 record=dict(root_id=rid,status='review' if flags else 'ok',flags=flags,coordinate_unit='um',area_unit='um2',source_coordinate_unit='nm',nm_to_um=.001,mesh_lod=0,mesh_source='https://storage.googleapis.com/flywire_v141_m783/mesh_mip_1_err_40',skeleton_source=skurl,skeleton_sha256=sha(skpath),mesh_arrays_sha256=sha(meshpath),absolute_mesh_vertices_sha256=fingerprint,sample_sha256=sha(sample),sample_bytes=sample.stat().st_size,coverage=coverage,raw_bytes=skpath.stat().st_size+meshpath.stat().st_size,skeleton_nodes=n,skeleton_edges=ne,mesh_vertices=len(v),mesh_faces=len(f),skeleton_outside_bbox_fraction=outside,skel_to_mesh_vertex_um={str(q):float(np.quantile(distance,q)) for q in (.5,.95,.99)},**view_audit)
 save(marker,record);return record

def run(root,mode,limit):
 root.mkdir(parents=True,exist_ok=True);lock=open(root/'build.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=[json.loads(line) for line in (root/'metadata/manifest_provisional.jsonl').read_text().splitlines()];all_rows=rows
 if mode=='pilot':
  rows=[min((r for r in rows if r['super_class']==c),key=lambda r:r['root_id']) for c in sorted({r['super_class'] for r in rows})]
 if limit:rows=rows[:limit]
 if mode=='full':assert (root/'pilot_contract.json').exists() and json.loads((root/'pilot_contract.json').read_text())['passed']
 cfg=root/'metadata/configs/build.json';assert json.loads(cfg.read_text())['workers']==1
 # Save public source format metadata once for provenance; no authentication is needed.
 for name,url in [('volume_info','https://storage.googleapis.com/flywire_v141_m783/info'),('mesh_info','https://storage.googleapis.com/flywire_v141_m783/mesh_mip_1_err_40/info')]:
  path=root/'metadata'/(name+'.json')
  if not path.exists():save(path,json.loads(get(url)))
 start=time.time();records=[]
 bytes_used=sum(p.stat().st_size for directory in ('raw','samples') for p in (root/directory).rglob('*') if p.is_file())
 for i,row in enumerate(rows):
  def root_bytes():
   paths=list((root/'raw'/row['root_id']).glob('*'))+[root/'samples'/(row['root_id']+'.npz')]
   return sum(p.stat().st_size for p in paths if p.is_file())
  before=root_bytes()
  assert bytes_used<60*2**30,'Dataset disk budget60GiB exceeded: inspect instead of deleting'
  error=None
  for attempt in range(3):
   try:rec=one(row,root);break
   except Exception as exc:
    error=repr(exc)
    if 'Disk' in error or 'disk budget' in error:raise
    if attempt<2:time.sleep(2*(attempt+1))
  else:
   rec=dict(root_id=row['root_id'],status='failed',error=error);save(root/'qa'/(row['root_id']+'.json'),rec)
  bytes_used+=root_bytes()-before
  records.append(rec);counts={k:sum(r['status']==k for r in records) for k in ('ok','review','failed')}
  save(root/(mode+'_status.json'),dict(state='building',done=i+1,total=len(rows),counts=counts,seconds=time.time()-start,last_root=row['root_id'],dataset_bytes=bytes_used));print(mode,i+1,len(rows),row['root_id'],rec['status'],flush=True)
 save(root/(mode+'_build.json'),dict(complete=True,mode=mode,candidate_count=len(all_rows),processed=len(rows),records=records,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},seconds=time.time()-start))
 save(root/(mode+'_status.json'),dict(state='complete' if all(r['status']=='ok' for r in records) else 'completed_with_exclusions',done=len(rows),total=len(rows),counts={k:sum(r['status']==k for r in records) for k in ('ok','review','failed')},seconds=time.time()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['pilot','full'],required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 try:run(a.root,a.mode,a.limit)
 except BaseException as e:save(a.root/(a.mode+'_failure.json'),dict(error=repr(e)));raise

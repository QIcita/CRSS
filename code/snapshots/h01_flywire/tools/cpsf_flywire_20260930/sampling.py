"""Surface-defined view supports, common to all modality controls."""
import numpy as np
from tools.cpsf_v19_measure_core.global_task import skeleton_thumbnail

def area_partition(centers,area,k):
 """Partition whole faces, not duplicated halos; labels and skeleton never enter."""
 groups=[np.arange(len(area))]
 while len(groups)<k:
  eligible=[i for i,g in enumerate(groups) if len(g)>1]
  if not eligible:break
  j=max(eligible,key=lambda i:(float(area[groups[i]].sum()),-i));g=groups.pop(j);xyz=centers[g];axis=int(np.ptp(xyz,axis=0).argmax());order=np.argsort(xyz[:,axis],kind='stable');g=g[order];cum=np.cumsum(area[g]);cut=int(np.searchsorted(cum,cum[-1]/2))+1;cut=max(1,min(cut,len(g)-1));groups[j:j]=[g[:cut],g[cut:]]
 return groups

def draw(vertices,faces,area,ids,n,rng):
 total=float(area[ids].sum());chosen=rng.choice(ids,size=n,replace=True,p=area[ids]/total);tri=vertices[faces[chosen]]
 u=np.sqrt(rng.random(n));v=rng.random(n);points=(1-u[:,None])*tri[:,0]+(u*(1-v))[:,None]*tri[:,1]+(u*v)[:,None]*tri[:,2]
 return points.astype(np.float32),np.full(n,total/n,np.float32),chosen.astype(np.int64)

def build_views(s,e,v,f,seed=42):
 s=np.asarray(s,np.float64);e=np.asarray(e,np.int64);v=np.asarray(v,np.float64);f=np.asarray(f,np.int64)
 tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 valid=np.flatnonzero(areas>1e-12);assert len(valid)>0
 centers=tri.mean(1);total=float(areas.sum());k=min(32,max(1,int(np.ceil(total/256.))))
 partitions=[valid[g] for g in area_partition(centers[valid],areas[valid],k)]
 assert np.array_equal(np.sort(np.concatenate(partitions)),valid)
 rng=np.random.default_rng(seed);result={};records=[];thumbnail_s,thumbnail_e=skeleton_thumbnail(s,e,2048)
 for i,ids in enumerate([valid]+partitions):
  points,weight,faceids=draw(v,f,areas,ids,4096 if i==0 else 2048,rng)
  if i==0:ss,ee=thumbnail_s,thumbnail_e
  else:
   bounds=v[np.unique(f[ids])];lo=bounds.min(0)-4.;hi=bounds.max(0)+4.
   # Conservative edge-AABB halo, retaining only observed edges; never heal a graph.
   edge_lo=np.minimum(s[e[:,0]],s[e[:,1]]);edge_hi=np.maximum(s[e[:,0]],s[e[:,1]])
   eligible=np.flatnonzero(((edge_hi>=lo)&(edge_lo<=hi)).all(1));nodeids=np.union1d(np.unique(e[eligible]),np.flatnonzero(((s>=lo)&(s<=hi)).all(1)))
   lut=np.full(len(s),-1,np.int64);lut[nodeids]=np.arange(len(nodeids));ss,ee=skeleton_thumbnail(s[nodeids],lut[e[eligible]],512)
  prefix=f'v{i:02d}_';result.update({prefix+'skeleton_xyz_um':ss.astype(np.float32),prefix+'edge_index':ee.astype(np.int64),prefix+'surface_xyz_um':points,prefix+'surface_weight_um2':weight,prefix+'source_face_id':faceids})
  records.append(dict(index=i,role='thumbnail' if i==0 else 'tile',area_um2=float(areas[ids].sum()),faces=len(ids),nodes=len(ss),edges=len(ee),points=len(points),skeleton_empty=not len(ss)))
 result['view_count']=np.asarray(len(records),np.int32)
 return result,dict(total_area_um2=total,nonzero_faces=len(valid),degenerate_faces=len(f)-len(valid),tiles=len(partitions),views=records,tile_area_sum_um2=sum(r['area_um2'] for r in records[1:]),partition_coverage=True)

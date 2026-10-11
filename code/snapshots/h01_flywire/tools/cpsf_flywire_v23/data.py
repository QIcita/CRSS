import gzip,io,json,struct,hashlib
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import sha,save
from tools.cpsf_h01_topology_repair.adapter import geometry
from tools.cpsf_v22_shared_region.data import regions

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/root/megrez-tmp/CPSF_FLYWIRE_TASKS_20260930')

def read_dataset():
    ready=json.loads((DATA/'dataset/READY.json').read_text())
    assert ready['ready'] and ready['seed']==42 and ready['duplicate_absolute_geometry_checked']
    assert sha(DATA/'dataset/frozen_manifest.jsonl')==ready['manifest_sha256']
    assert sha(DATA/'dataset/labels.json')==ready['labels_sha256']
    rows=[json.loads(l) for l in (DATA/'dataset/frozen_manifest.jsonl').read_text().splitlines()]
    assert len({r['root_id'] for r in rows})==len(rows)==8121
    return rows,json.loads((DATA/'dataset/labels.json').read_text()),ready

def skeleton(rid,qa):
    p=DATA/'raw'/rid/'skeleton.bin';assert sha(p)==qa['skeleton_sha256']
    b=p.read_bytes();n,ne=struct.unpack('<II',b[:8]);assert len(b)==8+16*n+8*ne
    s=np.frombuffer(b,'<f4',3*n,8).reshape(n,3).astype(np.float64)/1000
    e=np.frombuffer(b,'<u4',2*ne,8+12*n).reshape(ne,2).astype(np.int64)
    return s,e

def fixed_origin(g,s,e,p,w,origin):
    # Identical grid origin for all surface sampling variants; no label inputs.
    p,inv=np.unique(np.asarray(p,np.float64)-origin,axis=0,return_inverse=True)
    weight=np.bincount(inv,weights=w);_,owner=np.unique(np.floor(p/.5).astype(np.int64),axis=0,return_inverse=True)
    mass=np.bincount(owner,weights=weight);centers=np.zeros((len(mass),3))
    np.add.at(centers,owner,p*weight[:,None]);centers/=mass[:,None]
    rel=(p-centers[owner])/.5
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32)
    g.update(s=tensor(s-origin),p=tensor(p),w=tensor(weight),owner=torch.as_tensor(owner,dtype=torch.long),
        mass=tensor(mass),c=tensor(centers),pf=tensor(np.column_stack((rel,np.linalg.norm(rel,axis=1)))),
        origin=origin.tolist(),m=len(mass))
    return g

def uniform_draw(rid,qa,center,n):
    p=DATA/'raw'/rid/'mesh.npz';assert sha(p)==qa['mesh_arrays_sha256']
    with np.load(p) as x:v=x['vertices_nm'].astype(np.float64)/1000-center;f=x['faces']
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    ids=np.flatnonzero(areas>1e-12);rng=np.random.default_rng(42);total=float(areas[ids].sum())
    chosen=rng.choice(ids,n,replace=True,p=areas[ids]/total);tri=tri[chosen]
    u=np.sqrt(rng.random(n));z=rng.random(n)
    points=((1-u[:,None])*tri[:,0]+(u*(1-z))[:,None]*tri[:,1]+(u*z)[:,None]*tri[:,2]).astype(np.float32)
    return points,np.full(n,total/n,np.float32)

def build(row,mode):
    rid=row['root_id'];qa=json.loads((DATA/'qa'/(rid+'.json')).read_text());assert qa['status']=='ok'
    if mode=='skeleton':
        # Do not open surface samples, mesh arrays, or surface-defined supports.
        s,e=skeleton(rid,qa);s=s-s.mean(0)
        g=geometry(s,e,np.empty((0,3)));g['tile_count']=0
    else:
        p=DATA/'samples'/(rid+'.npz');assert sha(p)==row['sample_sha256']
        with np.load(p) as x:
            count=int(x['view_count']);points=[];weights=[]
            for i in range(count):
                points.append(x[f'v{i:02d}_surface_xyz_um']);weights.append(x[f'v{i:02d}_surface_weight_um2']*.5)
            full=np.concatenate(points);allweights=np.concatenate(weights)
            origin=(full.min(0).astype(float)+full.max(0).astype(float))/2
            if mode=='point':s=np.empty((0,3));e=np.empty((0,2),int)
            else:s,e=skeleton(rid,qa);s-=x['source_center_um']
            if mode=='thumbnail':p=points[0];w=weights[0]*2
            elif mode=='uniform_budget':p,w=uniform_draw(rid,qa,x['source_center_um'],len(full))
            else:p,w=full,allweights
        g=geometry(s,e,p,w);g=fixed_origin(g,s,e,p,w,origin)
        g['tile_count']=count-1 if mode in ('joint','point') else 0
        assert abs(float(g['mass'].sum())-qa['total_area_um2'])/qa['total_area_um2']<2e-5
    data=regions(g,mode if mode in ('point','skeleton') else 'joint')
    if mode!='point':assert data['raw_nodes']==qa['skeleton_nodes'] and data['raw_edges']==qa['skeleton_edges']
    if mode=='point':assert not data['has_g'].any() and data['raw_nodes']==0
    if mode=='skeleton':assert not data['has_p'].any() and data['patches']==0
    return data

def store(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as f:torch.save(data,f)
    temp.replace(path)

def load(path):
    with gzip.open(path,'rb') as f:return torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=False)

def record(row,mode,path,data):
    return dict(root_id=row['root_id'],split=row['split'],cell_type=row['cell_type'],
        coarse_target=row['coarse_target'],fine_target=row['fine_target'],variant=mode,
        path=str(path),sha256=sha(path),bytes=path.stat().st_size,nregions=data['nregions'],
        nodes=data['raw_nodes'],edges=data['raw_edges'],points=data['surface_points'])

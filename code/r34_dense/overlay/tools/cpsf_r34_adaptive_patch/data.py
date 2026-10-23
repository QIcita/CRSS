"""Same point coordinates/mass and skeleton; model-private patch organization."""
import gzip,hashlib,os,fcntl
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read,save,sha
from tools.cpsf_raw_point_attention.data import grouping
from tools.cpsf_r28_expansion.data import load_item as parent_load
from tools.cpsf_v22_shared_region.data import regions
from tools.cpsf_v24_physical_conditioning.data import adapt
from tools.cpsf_utonia_context.data import move

def fingerprint(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def statistics(owner):
    n=np.bincount(owner)
    return dict(patches=len(n),mean=float(n.mean()),median=float(np.median(n)),maximum=int(n.max()),
                singletons=float((n==1).mean()),patches_below_8=float((n<8).mean()),
                points_in_patches_below_8=float(n[n<8].sum()/n.sum()))

def context_fields(g,pos,w):
    # Utonia's own physical voxelization stays at 0.5. Only the per-point local
    # descriptor changes with the CMM grouping; never recover XYZ with a stale scale.
    origin=pos.min(0);grid=np.floor((pos-origin)/.5+1e-5).astype(np.int64)
    unique,inverse=np.unique(grid,axis=0,return_inverse=True)
    mass=np.bincount(inverse,weights=w,minlength=len(unique))
    pf=g['pf'].numpy().astype(np.float64);mu=np.zeros((len(unique),4));second=mu.copy();coord=np.zeros((len(unique),3))
    np.add.at(mu,inverse,pf*w[:,None]);np.add.at(second,inverse,pf**2*w[:,None]);np.add.at(coord,inverse,(pos-origin)*w[:,None])
    mu/=mass[:,None];second/=mass[:,None];coord/=mass[:,None]
    g.update(ucoord=torch.tensor(coord/.5,dtype=torch.float32),ugrid=torch.from_numpy(unique),uinverse=torch.from_numpy(inverse),
             ufeat=torch.tensor(np.column_stack((mu,(second-mu**2).clip(0),np.log1p(mass))),dtype=torch.float32))

def build(row):
    assert row['task'] in ('h01_global','flywire_coarse')
    for f,h in [('path','sha256'),('feature_path','feature_sha256'),('graph_path','graph_sha256')]:assert sha(row[f])==row[h],row[f]
    opener=gzip.open if row['path'].endswith('.gz') else open
    with opener(row['path'],'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=False)
    old=raw.get('g',raw)
    with np.load(row['feature_path']) as a:old_centres=a['patch_pos'].astype(np.float64)
    pos=old_centres[old['point_owner'].numpy()]+old['pf'][:,:3].numpy().astype(np.float64)*.5
    w=old['point_weight'].numpy().astype(np.float64);owner=old['point_owner'].numpy();before=statistics(owner)
    h=.5
    for level in range(25):
        h=.5*2**level
        if level:_,owner=np.unique(np.floor(pos/h).astype(np.int64),axis=0,return_inverse=True)
        if len(pos)/int(owner.max()+1)>=10:break
    else:raise RuntimeError('Unable to reach mean 10 within finite scale search')
    if h==.5:
        g=parent_load(row)['g']
        g.update(patch_size=.5,context_voxel_size=.5)
        return dict(g=g,audit=dict(patch_size=h,before=before,after=before,unchanged_parent_path=True,
             coordinate_sha256=fingerprint(pos),weight_sha256=fingerprint(w),point_count=len(pos),coordinate_error=0.,mass_error=0.))
    mass=np.bincount(owner,weights=w);centres=np.zeros((len(mass),3))
    np.add.at(centres,owner,pos*w[:,None]);centres/=mass[:,None]
    rel=(pos-centres[owner])/h
    with np.load(row['graph_path']) as a:
        edge=np.unique(np.sort(a['edge_index'].astype(np.int64),axis=1),axis=0)
        coord=a['coord'].astype(np.float64)
    assert len(coord)==len(old['s']) and np.allclose(coord-coord[0],old['s'].numpy()-old['s'][0].numpy(),atol=6e-4)
    assert len(edge)==old['raw_edges']
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32)
    geo=dict(s=old['s'],sf=old['sf'],e=torch.from_numpy(edge),n=len(old['s']),p=tensor(pos),w=old['point_weight'],
             owner=torch.from_numpy(owner),mass=tensor(mass),c=tensor(centres),pf=tensor(np.column_stack((rel,np.linalg.norm(rel,axis=1)))),
             m=len(mass),config=dict(voxel=h,radius=float(old['radius'])),origin=old['origin'],source_points=len(pos),tile_count=0)
    g=adapt(regions(geo,'joint'))
    # Real geometry quantization is bounded; FPS observations themselves are unchanged.
    reconstructed=centres[owner]+g['pf'][:,:3].numpy().astype(np.float64)*h
    error=float(np.abs(reconstructed-pos).max())
    assert error<5e-5 and np.isclose(g['patch_mass'].sum().item(),w.sum(),rtol=3e-5)
    assert torch.equal(g['s'],old['s']) and torch.equal(g['sf'],old['sf'])
    for k in ['sr','sn','sw']:assert torch.equal(g[k],old[k]),('skeleton support drift',k)
    assert torch.equal(g['graph_mass'][g['has_g']],old['graph_mass'][old['has_g']])
    # Surface descriptors can influence multiple observed skeleton regions; keep
    # original soft associations and explicit orphan tokens, never hard-drop points.
    den=np.bincount(g['pj'].numpy(),weights=g['pw'].numpy(),minlength=g['patches'])
    assert np.allclose(den,1,rtol=3e-5,atol=3e-5)
    context_fields(g,pos,w)
    g.update(grouping(owner,len(mass)),raw_patch_pos=tensor(centres/h),patch_size=h,context_voxel_size=.5)
    assert not {'target','label','split','observed','fine_target','coarse_target'}.intersection(g)
    return dict(g=g,audit=dict(patch_size=h,before=before,after=statistics(owner),unchanged_parent_path=False,
          coordinate_sha256=fingerprint(pos),weight_sha256=fingerprint(w),point_count=len(pos),coordinate_error=error,
          mass_error=float(abs(g['patch_mass'].sum().item()-w.sum())),skeleton_support_exact=True,
          regions=g['nregions'],incidences=len(g['pr']),orphans=int((~g['has_g']).sum()),
          context_voxel_size=.5,context_tokens=len(g['ufeat']),all_source_points_used=True))

class Cache:
    def __init__(self,force_parent=False):self.items=OrderedDict();self.size=0;self.force_parent=force_parent
    def get(self,row):
        key=(row['task'],row['key'])
        if key in self.items:v,n=self.items.pop(key);self.items[key]=(v,n);return v
        if self.force_parent:
            v=parent_load(row);v['g']['patch_size']=.5
        else:
            folder=ROOT/'adaptive_cache'/row['task'];folder.mkdir(parents=True,exist_ok=True)
            path=folder/(row['key']+'.pt');meta=path.with_suffix('.json')
            with open(path.with_suffix('.lock'),'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not meta.exists():
                    import shutil
                    assert shutil.disk_usage(ROOT).free>6*2**30,'Keep at least 6 GiB disk free'
                    v=build(row);temp=path.with_suffix('.tmp');torch.save(v,temp);temp.replace(path)
                    save(meta,dict(source_sha256=row['sha256'],feature_sha256=row['feature_sha256'],
                        protocol_sha256=sha(ROOT/'protocol.json'),cache_sha256=sha(path),audit=v['audit']))
                else:
                    m=read(meta)
                    assert m['source_sha256']==row['sha256'] and m['feature_sha256']==row['feature_sha256']
                    assert m['protocol_sha256']==sha(ROOT/'protocol.json') and m['cache_sha256']==sha(path)
                    v=torch.load(path,map_location='cpu',weights_only=False)
        n=sum(t.numel()*t.element_size() for t in v['g'].values() if isinstance(t,torch.Tensor))
        while self.items and self.size+n>768*2**20:
            _,(_,size)=self.items.popitem(last=False);self.size-=size
        if n<768*2**20:self.items[key]=(v,n);self.size+=n
        return v

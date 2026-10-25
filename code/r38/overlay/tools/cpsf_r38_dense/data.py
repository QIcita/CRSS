"""Same AP10 computation groups; no new fine support or upstream input."""
from collections import OrderedDict
import numpy as np
import torch
from .common import sha
from tools.cpsf_r28_expansion.data import load_item as parent_load
from tools.cpsf_raw_point_attention.data import grouping
from tools.cpsf_utonia_context.data import move

DENSE_FIELDS=('point_owner','patch_mass','patches','raw_patch_ids','raw_indices','raw_valid','raw_counts','raw_layout')

def build(row,force_parent=False):
    for f,h in [('path','sha256'),('feature_path','feature_sha256'),('graph_path','graph_sha256')]:
        assert sha(row[f])==row[h],row[f]
    original=parent_load(row)['g'];g=dict(original);g['patch_size']=.5
    owner=g['point_owner'].numpy();h=.5;n=len(owner);audit={'point_count':n,'fine_patches':g['patches']}
    if not force_parent and n/g['patches']<10:
        with np.load(row['feature_path']) as a:centers=a['patch_pos'].astype(np.float64)
        pos=centers[owner]+g['pf'][:,:3].numpy().astype(np.float64)*.5
        w=g['point_weight'].numpy().astype(np.float64)
        for level in range(1,25):
            h=.5*2**level
            _,dense_owner=np.unique(np.floor(pos/h).astype(np.int64),axis=0,return_inverse=True)
            if n/(dense_owner.max()+1)>=10:break
        else:raise RuntimeError('Finite scale search exhausted')
        mass=np.bincount(dense_owner,weights=w);c=np.zeros((len(mass),3))
        np.add.at(c,dense_owner,pos*w[:,None]);c/=mass[:,None]
        rel=(pos-c[dense_owner])/h
        # Match the successful R34 repair's float32 AP10->physical-unit arithmetic.
        pf=torch.tensor(np.column_stack((rel,np.linalg.norm(rel,axis=1))),dtype=torch.float32)*(h/.5)
        cscaled=torch.tensor(c/h,dtype=torch.float32)*(h/.5)
        dense=dict(point_owner=torch.from_numpy(dense_owner),patch_mass=torch.tensor(mass,dtype=torch.float32),patches=len(mass),**grouping(dense_owner,len(mass)))
        g.update({'dense_'+k:v for k,v in dense.items()},dense_pf=pf,dense_raw_patch_pos=cscaled,dense_size=h)
        reconstructed=cscaled[dense['point_owner']].numpy().astype(float)*.5+pf[:,:3].numpy().astype(float)*.5
        assert np.allclose(reconstructed,pos,rtol=1e-6,atol=5e-5)
        assert torch.allclose(dense['patch_mass'].sum(),g['patch_mass'].sum(),rtol=3e-5)
        counts=np.bincount(dense_owner)
        audit.update(dense_patches=len(mass),mean_dense_points=float(counts.mean()),max_dense_points=int(counts.max()),coordinate_error=float(abs(reconstructed-pos).max()))
    for k,v in original.items():
        if isinstance(v,torch.Tensor):assert torch.equal(g[k],v),('original input drift',k)
    assert not {'target','label','split','observed','fine_target','coarse_target'}.intersection(g)
    audit.update(neighborhood_size=h,original_fields_exact=True,force_parent=force_parent)
    return dict(g=g,audit=audit)

class Cache:
    def __init__(self,force_parent=False):self.items=OrderedDict();self.size=0;self.force_parent=force_parent
    def get(self,row):
        key=(row['task'],row['key'])
        if key in self.items:
            v,n=self.items.pop(key);self.items[key]=(v,n);return v
        v=build(row,self.force_parent)
        n=sum(x.numel()*x.element_size() for x in v['g'].values() if isinstance(x,torch.Tensor))
        while self.items and self.size+n>768*2**20:
            _,(_,size)=self.items.popitem(last=False);self.size-=size
        if n<768*2**20:self.items[key]=(v,n);self.size+=n
        return v

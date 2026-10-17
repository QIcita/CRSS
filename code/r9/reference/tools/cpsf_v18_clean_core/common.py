import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from collections import OrderedDict
import hashlib,json,random
import numpy as np
import torch
from .encoder import JointEmbedding

BASE=Path('/root/megrez-tmp')
SOURCE=BASE/'H01_HIGH_COVERAGE_FULL104_20260920'
V9=BASE/'CPSF_H01_V9_20260921/cache_local'
TED=BASE/'CPSF_SHARED_V4_20260919/cache_ted'
NREDD=BASE/'NREDD_proj/NREDD'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def setup(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.getenv('CPSF_GPU_FRACTION','.32')))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

class LRU:
    def __init__(self,budget=600*2**20,disk=None):
        self.items=OrderedDict();self.bytes=0;self.budget=budget
        self.disk=Path(disk) if disk else None
        if self.disk:self.disk.mkdir(parents=True,exist_ok=True)
    def get(self,key,make):
        if key in self.items:
            item,size=self.items.pop(key);self.items[key]=(item,size);return item
        disk_path=None
        if self.disk:
            import hashlib
            disk_path=self.disk/(hashlib.sha1(str(key).encode()).hexdigest()+'.pt')
            if disk_path.is_file():
                item=load(disk_path)
                size=sum(v.numel()*v.element_size() for v in item.values() if isinstance(v,torch.Tensor))
                if size<=self.budget:
                    while self.items and self.bytes+size>self.budget:
                        _,(_,old)=self.items.popitem(last=False);self.bytes-=old
                    self.items[key]=(item,size);self.bytes+=size
                return item
        item=make();size=sum(v.numel()*v.element_size() for v in item.values() if isinstance(v,torch.Tensor))
        if disk_path and not disk_path.exists():
            tmp=disk_path.with_suffix('.tmp');torch.save(item,tmp);tmp.replace(disk_path)
        while self.items and self.bytes+size>self.budget:
            _,(_,old)=self.items.popitem(last=False);self.bytes-=old
        if size<=self.budget:self.items[key]=(item,size);self.bytes+=size
        return item

def skeleton_view(g):
    out=dict(g);n=g['n_nodes'];keep=g['local_receiver']<n
    for key in ('point_features','point_xyz','point_patch','point_neighbors','neighbor_valid','patch_xyz',
                'patch_point_count','original_to_unique','cross_anchor','cross_surface',
                'segment_xyz','segment_features','cross_segment','cross_segment_surface'):
        out[key]=g[key][:0]
    out.update(local_receiver=g['local_receiver'][keep],local_sender=g['local_sender'][keep],
               positions=g['positions'][:n],n_patches=0,n_segments=0,input_point_count=0,unique_point_count=0)
    return out

def pack_geometry(gs,device='cuda'):
    """Disjoint union with independent normalization, local graphs and cross incidence."""
    ns=np.r_[0,np.cumsum([g['n_nodes'] for g in gs])];ps=np.r_[0,np.cumsum([g['n_patches'] for g in gs])]
    qs=np.r_[0,np.cumsum([g['n_segments'] for g in gs])]
    xs=np.r_[0,np.cumsum([len(g['point_xyz']) for g in gs])]
    n,m,q=int(ns[-1]),int(ps[-1]),int(qs[-1]);result={}
    for key in ('skeleton_features','skeleton_xyz','point_features','point_xyz','patch_xyz','patch_point_count',
                'segment_xyz','segment_features'):
        result[key]=torch.cat([g[key] for g in gs]).to(device)
    # Segment tokens feed back to their observed edge endpoints.  Preserve
    # that topology in the disjoint union with a node offset; omitting it
    # makes packed multi-sample inference fail only when the new segment path
    # is enabled.
    result['observed_edges']=torch.cat([g['observed_edges']+int(ns[i]) for i,g in enumerate(gs)]).to(device)
    result['positions']=torch.cat((result['skeleton_xyz'],result['patch_xyz']))
    result['point_patch']=torch.cat([g['point_patch']+int(ps[i]) for i,g in enumerate(gs)]).to(device)
    near=[];valid=[];k=max(g['point_neighbors'].shape[1] for g in gs)
    for i,g in enumerate(gs):
        idx=g['point_neighbors']+int(xs[i]);mask=g['neighbor_valid'];pad=k-idx.shape[1]
        near.append(torch.nn.functional.pad(idx,(0,pad),value=int(xs[i])));valid.append(torch.nn.functional.pad(mask,(0,pad),value=False))
    result['point_neighbors']=torch.cat(near).to(device);result['neighbor_valid']=torch.cat(valid).to(device)
    for key in ('local_receiver','local_sender'):
        result[key]=torch.cat([torch.where(g[key]<g['n_nodes'],g[key]+int(ns[i]),g[key]-g['n_nodes']+n+int(ps[i]))
                               for i,g in enumerate(gs)]).to(device)
    result['cross_anchor']=torch.cat([g['cross_anchor']+int(ns[i]) for i,g in enumerate(gs)]).to(device)
    result['cross_surface']=torch.cat([g['cross_surface']+int(ps[i]) for i,g in enumerate(gs)]).to(device)
    result['cross_segment']=torch.cat([g['cross_segment']+int(qs[i]) for i,g in enumerate(gs)]).to(device)
    result['cross_segment_surface']=torch.cat([g['cross_segment_surface']+int(ps[i]) for i,g in enumerate(gs)]).to(device)
    result.update(n_nodes=n,n_patches=m,n_segments=q,geometry_config=gs[0]['geometry_config'])
    assert all(g['geometry_config']==result['geometry_config'] for g in gs)
    return result,(ns,ps)

def unbind(z,offsets):
    ns,ps=offsets
    return [JointEmbedding(z.skeleton[a:b],z.surface[c:d],z.skeleton_xyz[a:b],z.surface_xyz[c:d],z.surface_mass[c:d])
            for a,b,c,d in zip(ns[:-1],ns[1:],ps[:-1],ps[1:])]

def classification(y,p):
    matrix=np.bincount(np.asarray(y)*3+np.asarray(p),minlength=9).reshape(3,3);tp=matrix.diagonal()
    f1=2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)
    return dict(macro_f1=float(f1.mean()),accuracy=float(tp.sum()/matrix.sum()),
        balanced_accuracy=float((tp/np.maximum(matrix.sum(1),1)).mean()),per_class_f1=f1.tolist(),confusion_matrix=matrix.tolist())

def regression(y,p):
    y=np.asarray(y);p=np.asarray(p);den=((y-y.mean())**2).sum()
    return dict(mae=float(abs(y-p).mean()),rmse=float(np.sqrt(((y-p)**2).mean())),
        r2=float(1-((y-p)**2).sum()/den) if den>0 else None,
        pearson=float(np.corrcoef(y,p)[0,1]) if min(y.std(),p.std())>1e-12 else None)

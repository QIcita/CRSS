import os,json,hashlib,random
from pathlib import Path
from collections import OrderedDict
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
BASE=Path('/root/megrez-tmp');SOURCE=BASE/'H01_HIGH_COVERAGE_FULL104_20260920'
V9=BASE/'CPSF_H01_V9_20260921/cache_local';TED=BASE/'CPSF_SHARED_V4_20260919/cache_ted'
NREDD=BASE/'NREDD_proj/NREDD'
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def setup():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(float(os.getenv('CPSF_GPU_FRACTION','.42')))
def classification(y,p):
 matrix=np.bincount(np.asarray(y)*3+np.asarray(p),minlength=9).reshape(3,3);tp=matrix.diagonal()
 return dict(macro_f1=float((2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)).mean()),accuracy=float(tp.sum()/matrix.sum()),per_class_f1=(2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)).tolist(),confusion_matrix=matrix.tolist())
def counts(y,p):
 y=np.asarray(y);p=np.asarray(p);tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum());pe=tp/(tp+fp+1e-5);re=tp/(tp+fn+1e-5)
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),released_epsilon_metrics=dict(precision=pe,recall=re,f1=2*pe*re/(pe+re+1e-5)))
class Cache:
 def __init__(self,budget=1024*2**20,disk=None):
  self.items=OrderedDict();self.bytes=0;self.budget=budget;self.disk=Path(disk) if disk else None
  if self.disk:self.disk.mkdir(parents=True,exist_ok=True)
 def get(self,k,make):
  if k in self.items:
   x,b=self.items.pop(k);self.items[k]=(x,b);return x
  path=self.disk/(hashlib.sha256(k.encode()).hexdigest()+'.pt') if self.disk else None
  if path is not None and path.exists():x=load(path)
  else:
   x=make()
   if path is not None:
    import shutil
    assert shutil.disk_usage(self.disk).free>4*2**30,'Preserve disk headroom; no automatic deletion'
    tmp=path.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(path)
  def size(v):
   if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
   if isinstance(v,dict):return sum(size(z) for z in v.values())
   if isinstance(v,(tuple,list)):return sum(size(z) for z in v)
   return 0
  b=size(x)
  while self.items and self.bytes+b>self.budget:
   _,(_,old)=self.items.popitem(last=False);self.bytes-=old
  if b<=self.budget:self.items[k]=(x,b);self.bytes+=b
  return x

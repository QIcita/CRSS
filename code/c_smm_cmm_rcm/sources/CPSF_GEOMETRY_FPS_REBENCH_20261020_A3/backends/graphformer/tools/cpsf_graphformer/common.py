import os,json,hashlib,random
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('GRAPHFORMER_ROOT',Path(__file__).resolve().parents[2]))
BASE=Path('/root/megrez-tmp')
TASKS=('h01_local','h01_global','flywire_coarse')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def setup():
 import torch
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(False)
def verify():
 p=json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,('source drift',path)
 for path,h in p['parent_sha256'].items():assert sha(path)==h,('parent drift',path)
 return p

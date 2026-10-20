"""Frozen R9 engineering, seed42 and independent artifact identity."""
import os,json,hashlib,random
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/root/megrez-tmp/CPSF_RAW_POINT_ATTENTION_20261019_R28_A1')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def setup():
    from tools.cpsf_utonia_context.kernel_policy import install
    install();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('CPSF_GPU_FRACTION','.45')))
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for k,h in p['source_sha256'].items():assert sha(ROOT/k)==h,('source drift',k)
    for k,h in p['parent_sha256'].items():assert sha(k)==h,('parent drift',k)
    return p
def rows(task):return json.loads((ROOT/'data'/task/'manifest.json').read_text())['records']
def load_item(row):
    from .data import load_item as impl
    return impl(row)
def state_sha(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

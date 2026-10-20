import gzip,io,json,os,hashlib
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp')
NREDD=BASE/'NREDD_proj/NREDD'
TED=BASE/'CPSF_SHARED_V4_20260919/cache_ted'
TLP=NREDD/'data/link_pred_data/official_192_81_geometry_v2e_cache'
ARMS=('full','mean_only','no_fusion','point','skeleton','no_position','no_scale','no_position_scale','shuffle_regions','no_context','thumbnail','uniform_budget')
TASKS=('ted','tlp')
def modal(arm):return arm if arm in ('point','skeleton','thumbnail','uniform_budget') else 'joint'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def store(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wb',compresslevel=1) as f:torch.save(obj,f)
    tmp.replace(path)
def load(path):
    with gzip.open(path,'rb') as f:return torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=False)
def move(g):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in g.items()}
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,('source drift',n)
    assert sha(ROOT/'split_manifest.json')==p['split_sha256']
    return p
def rows(task,arm):
    path=ROOT/'data'/task/modal(arm)/'manifest.json';d=json.loads(path.read_text())
    assert d['complete'] and d['protocol_sha256']==sha(ROOT/'protocol.json')
    return d['records'],sha(path)

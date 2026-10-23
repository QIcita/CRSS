import hashlib,json,os,sys
from pathlib import Path

BASE=Path('/root/megrez-tmp')
ROOT=Path(os.environ.get('CPSF_AP_ROOT',Path(__file__).resolve().parents[2]))
REFERENCE=BASE/'CPSF_R34_FPS_REBENCH_20261021_A2'
INPUT=BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
PYTHON=BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
TASKS=('h01_global','flywire_coarse')
REFERENCE_SHA='a0e71fa711f1f1713c8cc65102a220c7dbb387b60f4698e644c83f11614f4738'
INPUT_SHA='01bad09907e5540aa5bf15a9ca46f573bd3b23a0fb424ddcf270eccb851d7213'
sys.path.insert(0,str(INPUT/'deps'))
import tools
tools.__path__=list(dict.fromkeys([str(ROOT/'tools'),str(REFERENCE/'tools'),str(INPUT/'backends/r28/tools'),str(INPUT/'tools')]+list(tools.__path__)))

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(task):return read(INPUT/'data'/task/'manifest.json')['records']
def verify():
    p=read(ROOT/'protocol.json')
    assert sha(REFERENCE/'protocol.json')==REFERENCE_SHA and sha(INPUT/'protocol.json')==INPUT_SHA
    for f,h in p['files'].items():assert sha(f)==h,('source drift',f)
    for t,h in p['input_manifests'].items():assert sha(INPUT/'data'/t/'manifest.json')==h
    return p

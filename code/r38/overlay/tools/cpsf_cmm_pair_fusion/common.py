import hashlib,json,os,sys
from pathlib import Path

BASE=Path('/root/megrez-tmp')
ROOT=Path(os.environ.get('CPSF_CMM_PAIR_ROOT',Path(__file__).resolve().parents[2]))
PARENT=BASE/'CPSF_R34_FPS_REBENCH_20261021_A2'
GEOMETRY=BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
FINE=BASE/'CPSF_FLYWIRE_FINE_REBENCH_20261020_A1'
PYTHON=BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
PARENT_SHA='a0e71fa711f1f1713c8cc65102a220c7dbb387b60f4698e644c83f11614f4738'
TASKS=('h01_local','h01_global')
MODEL='r38'
PREFIX='cmm_pair_fusion.'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(task):return read(ROOT/'data'/task/'manifest.json')['records']
def add_backend():
    import tools
    extra=[str(PARENT/'tools'),str(GEOMETRY/'backends/r28/tools'),str(GEOMETRY/'tools')]
    tools.__path__=[str(x) for x in tools.__path__ if str(x) not in extra]+extra
    sys.path.insert(0,str(GEOMETRY/'deps'))
def verify():
    assert sha(PARENT/'protocol.json')==PARENT_SHA
    p=read(ROOT/'protocol.json')
    for f,h in p['files'].items():assert sha(f)==h,('source drift',f)
    return p
def runtime(gpu,task):
    env=dict(os.environ);env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CPSF_CMM_PAIR_ROOT=str(ROOT),
        CPSF_R34_FPS_ROOT=str(PARENT),CPSF_REBENCH_ROOT=str(GEOMETRY),
        CPSF_GPU_FRACTION='.08' if task=='h01_local' else '.36',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(GEOMETRY/'deps'))
    return env

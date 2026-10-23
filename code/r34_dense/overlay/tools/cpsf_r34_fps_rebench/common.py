import hashlib,json,os,sys
from pathlib import Path

BASE=Path('/root/megrez-tmp')
PARENT=BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
FINE=BASE/'CPSF_FLYWIRE_FINE_REBENCH_20261020_A1'
OLD_R34=BASE/'CPSF_PATH_MESSAGE_20261020_R34_A2'
ROOT=Path(os.environ.get('CPSF_R34_FPS_ROOT',Path(__file__).resolve().parents[2]))
PYTHON=BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
PARENT_SHA='01bad09907e5540aa5bf15a9ca46f573bd3b23a0fb424ddcf270eccb851d7213'
FINE_SHA='fca78e62648ed045966c4256f2487c3479508297454df9e235537528a04cdf47'
OLD_SHA='eaf723615ae22df382e9db542a9c4bed94ccafe947d0378c221273641c4931e3'
TASKS=('h01_local','h01_global','flywire_coarse','flywire_fine','ted','tlp')
sys.path.insert(0,str(PARENT/'deps'))

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def parent_for(task):return FINE if task=='flywire_fine' else PARENT
def manifest(task):return parent_for(task)/'data'/task/'manifest.json'
def rows(task):return read(manifest(task))['records']
def add_backend():
    import tools
    # Materialize namespace paths: the inherited common module inserts deps in
    # sys.path, which otherwise recomputes _NamespacePath and drops appends.
    extra=[str(PARENT/'backends/r28/tools'),str(PARENT/'tools')]
    tools.__path__=[str(x) for x in tools.__path__ if str(x) not in extra]+extra
def verify():
    assert sha(PARENT/'protocol.json')==PARENT_SHA
    assert sha(FINE/'protocol.json')==FINE_SHA
    assert sha(OLD_R34/'protocol.json')==OLD_SHA
    p=read(ROOT/'protocol.json')
    for f,h in p['files'].items():assert sha(f)==h,('source drift',f)
    return p
def runtime(gpu):
    env=dict(os.environ);env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CPSF_R34_FPS_ROOT=str(ROOT),CPSF_REBENCH_ROOT=str(PARENT),
        CPSF_GPU_FRACTION='.90',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(PARENT/'deps'))
    return env

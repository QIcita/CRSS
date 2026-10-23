import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get('CPSF_REBENCH_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT/'deps'))
BASE = Path('/root/megrez-tmp')
R28 = BASE/'CPSF_R28_TASK_EXPANSION_20261020_A2'
PYTHON = BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
TASKS = ('h01_local','h01_global','flywire_coarse','ted','tlp')
SOURCES = dict(r28=R28, phase2=BASE/'CPSF_EXTERNAL_BASELINES_PHASE2_20261010',
    phase3=BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012',
    utonia=BASE/'CPSF_UTONIA_SCRATCH_R2_20261012',
    graphformer=BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013',
    nredd=BASE/'CPSF_NREDD_BASELINES_20261014')
FAMILIES = dict(r28='r28',ptv3='phase2',litept='phase2',octformer='phase2',
    sgformer='phase3',polynormer='phase3',utonia='utonia',graphformer='graphformer',
    tednet='nredd',tlpnet='nredd')

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');q.replace(p)
def source_rows(task): return read(R28/'data'/task/'manifest.json')['records']
def rows(task): return read(ROOT/'data'/task/'manifest.json')['records']
def add_backend(family):
    import tools
    p=ROOT/'backends'/family/'tools'
    tools.__path__=[str(p)]+[str(x) for x in tools.__path__ if str(x)!=str(p)]
def verify():
    p=read(ROOT/'protocol.json')
    for f,h in p['files'].items():
        assert sha(f)==h,('source drift',f)
    return p
def runtime(gpu):
    env=dict(os.environ);env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CPSF_REBENCH_ROOT=str(ROOT),
        CPSF_GPU_FRACTION='.90',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT/'deps'))
    return env

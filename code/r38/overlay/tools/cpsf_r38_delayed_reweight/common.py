import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path('/root/megrez-tmp')
ROOT = Path(os.environ.get('CPSF_DRW_ROOT', Path(__file__).resolve().parents[2]))
DENSE = BASE/'CPSF_R38_DENSE_SUPPORT_20261023_A1'
PAIR = BASE/'CPSF_CMM_PAIR_FUSION_20261023_R38_A1'
R34 = BASE/'CPSF_R34_FPS_REBENCH_20261021_A2'
INPUT = BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
AP = BASE/'CPSF_R34_ADAPTIVE_PATCH10_20261022_A1'
PYTHON = BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
TASK = 'flywire_coarse'
DENSE_SHA = '1998e9e8a147855288e483732124e6678f9f0765c79f3f35b343c95f92016226'
LABELS = ['ascending', 'central', 'optic', 'sensory', 'visual_centrifugal', 'visual_projection']

sys.path.insert(0, str(INPUT/'deps'))
import tools
paths = [ROOT/'tools', DENSE/'tools', PAIR/'tools', R34/'tools', AP/'tools', INPUT/'backends/r28/tools', INPUT/'tools']
tools.__path__ = list(dict.fromkeys([str(p) for p in paths]+list(tools.__path__)))

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(p)
def rows(task):
    assert task == TASK
    return read(INPUT/'data'/task/'manifest.json')['records']
def verify():
    assert sha(DENSE/'protocol.json') == DENSE_SHA
    p = read(ROOT/'protocol.json')
    for f, h in p['files'].items(): assert sha(f) == h, ('source drift', f)
    assert sha(INPUT/'data'/TASK/'manifest.json') == p['input_manifests'][TASK]
    assert p['tasks'][TASK] == read(DENSE/'protocol.json')['tasks'][TASK]
    assert p['primary_metric'] == 'macro_f1' and p['loss']['alpha'] == .25
    return p
def runtime(gpu=0):
    env = dict(os.environ)
    env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CPSF_DRW_ROOT=str(ROOT),
               CPSF_R38_DENSE_ROOT=str(DENSE), CPSF_CMM_PAIR_ROOT=str(PAIR),
               CPSF_R34_FPS_ROOT=str(R34), CPSF_REBENCH_ROOT=str(INPUT),
               PYTHONPATH=str(INPUT/'deps'), CPSF_GPU_FRACTION='.85',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env

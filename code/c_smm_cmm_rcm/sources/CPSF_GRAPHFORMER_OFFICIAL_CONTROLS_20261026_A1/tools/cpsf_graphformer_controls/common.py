import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path('/root/megrez-tmp')
ROOT = Path(os.environ.get('GRAPHFORMER_CONTROLS_ROOT', BASE/'CPSF_GRAPHFORMER_OFFICIAL_CONTROLS_20261026_A1'))
PARENT = BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
FROZEN = PARENT/'backends/graphformer'
ORIGINAL = BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013'
PYTHON = BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
ARMS = ('official_pi_only', 'official_features_head')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def backends():
    import tools
    os.environ['GRAPHFORMER_ROOT'] = str(FROZEN)
    tools.__path__ = list(dict.fromkeys(list(tools.__path__)+[str(FROZEN/'tools')]))
    sys.path.insert(0, str(PARENT/'deps'))


def setup():
    backends()
    from tools.cpsf_graphformer.common import setup as original_setup
    original_setup()
    import torch
    if os.environ.get('CONTROLS_GPU_LIMIT_MIB'):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95, int(os.environ['CONTROLS_GPU_LIMIT_MIB'])*2**20/total))


def runtime(gpu, reserve):
    env = dict(os.environ)
    env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(GRAPHFORMER_CONTROLS_ROOT=str(ROOT), GRAPHFORMER_ROOT=str(FROZEN),
               CUDA_VISIBLE_DEVICES=str(gpu), CONTROLS_GPU_LIMIT_MIB=str(reserve),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(PARENT/'deps'))
    return env


def verify():
    p = read(ROOT/'protocol.json')
    for path, expected in p['files'].items():
        assert sha(path) == expected, ('frozen source/data drift', path)
    return p

import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path('/root/megrez-tmp')
ROOT = Path(os.environ.get('MICRONS_ROOT', BASE / 'CPSF_MICRONS_INPUT_20261025_A1'))
PYTHON = BASE / 'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
INPUT = BASE / 'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
DENSE = BASE / 'CPSF_R38_DENSE_SUPPORT_20261023_A1'
PARENTS = [DENSE, BASE / 'CPSF_CMM_PAIR_FUSION_20261023_R38_A1',
           BASE / 'CPSF_R34_FPS_REBENCH_20261021_A2',
           BASE / 'CPSF_R34_ADAPTIVE_PATCH10_20261022_A1',
           INPUT / 'backends/r28', INPUT / 'backends/utonia',
           INPUT / 'backends/phase3', INPUT]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def backends():
    sys.path.insert(0, str(INPUT / 'deps'))
    import tools
    tools.__path__ = list(dict.fromkeys([str(ROOT / 'tools')] +
                         [str(p / 'tools') for p in PARENTS] + list(tools.__path__)))


def runtime(gpu):
    env = dict(os.environ)
    env.update(read(BASE / 'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MICRONS_ROOT=str(ROOT),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(INPUT / 'deps'))
    return env


def seed():
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify():
    p = read(ROOT / 'protocol.json')
    for path, expected in p['files'].items():
        assert sha(path) == expected, ('source drift', path)
    assert sha(ROOT / 'cohort.json') == p['cohort_sha256']
    return p

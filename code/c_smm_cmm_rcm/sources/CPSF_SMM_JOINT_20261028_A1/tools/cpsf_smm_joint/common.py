import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path('/root/megrez-tmp')
ROOT = BASE/'CPSF_SMM_JOINT_20261028_A1'
PARENT = BASE/'CPSF_TOPOLOGY_SURFACE_20261028_A1'
INPUT = BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
CONTROLS = BASE/'CPSF_GRAPHFORMER_OFFICIAL_CONTROLS_20261026_A1'
DENSE = BASE/'CPSF_R38_DENSE_SUPPORT_20261023_A1'
JOINT = BASE/'CPSF_R38_RCM_REFACTOR_20261026_A1'
PYTHON = BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
ARMS = ('smm_joint', 'surface_conditioned', 'surface_mean')


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def paths():
    import tools
    os.environ['GRAPHFORMER_ROOT'] = str(INPUT/'backends/graphformer')
    os.environ['GRAPHFORMER_CONTROLS_ROOT'] = str(CONTROLS)
    os.environ['CPSF_R38_DENSE_ROOT'] = str(DENSE)
    sys.path.insert(0, str(INPUT/'deps'))
    roots = [ROOT, PARENT, JOINT, DENSE, CONTROLS, BASE/'CPSF_CMM_PAIR_FUSION_20261023_R38_A1',
             BASE/'CPSF_R34_FPS_REBENCH_20261021_A2', BASE/'CPSF_R34_ADAPTIVE_PATCH10_20261022_A1',
             INPUT/'backends/r28', INPUT/'backends/graphformer', INPUT]
    tools.__path__ = list(dict.fromkeys([str(p/'tools') for p in roots]+list(tools.__path__)))


def setup():
    paths()
    from tools.cpsf_utonia_context.kernel_policy import install
    from tools.cpsf_graphformer.common import setup as seed
    import torch
    install(); seed()
    # Same numerical policy as the strong GraPHFormer control.
    torch.cuda.set_per_process_memory_fraction(int(os.environ.get('TOPOLOGY_GPU_MIB', '16000'))*2**20/torch.cuda.get_device_properties(0).total_memory)


def runtime(gpu=0, reserve=16000):
    env = dict(os.environ); env.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TOPOLOGY_GPU_MIB=str(reserve),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(INPUT/'deps'))
    return env


def verify():
    p = read(ROOT/'protocol.json')
    for path, expected in p['files'].items():
        assert sha(path) == expected, ('Frozen source drift', path)
    return p


paths()

import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path('/root/megrez-tmp')
ROOT = Path(os.environ.get('CPSF_R38_DENSE_EXTENSION_ROOT', Path(__file__).resolve().parents[2]))
DENSE = BASE / 'CPSF_R38_DENSE_SUPPORT_20261023_A1'
R38 = BASE / 'CPSF_CMM_PAIR_FUSION_20261023_R38_A1'
R34 = BASE / 'CPSF_R34_FPS_REBENCH_20261021_A2'
INPUT = BASE / 'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
FINE = BASE / 'CPSF_FLYWIRE_FINE_REBENCH_20261020_A1'
AP = BASE / 'CPSF_R34_ADAPTIVE_PATCH10_20261022_A1'
PYTHON = BASE / 'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
TASKS = ('flywire_fine',)
PARENTS = {
    DENSE: '1998e9e8a147855288e483732124e6678f9f0765c79f3f35b343c95f92016226',
    R38: 'f09842b051cefd9bc8825357608a142566521fbdda368c626ee0d1af79ce900e',
    R34: 'a0e71fa711f1f1713c8cc65102a220c7dbb387b60f4698e644c83f11614f4738',
    INPUT: '01bad09907e5540aa5bf15a9ca46f573bd3b23a0fb424ddcf270eccb851d7213',
    FINE: 'fca78e62648ed045966c4256f2487c3479508297454df9e235537528a04cdf47',
}
sys.path.insert(0, str(INPUT / 'deps'))
import tools
tools.__path__ = list(dict.fromkeys(
    [str(p / 'tools') for p in (ROOT, DENSE, R38, R34, AP, INPUT / 'backends/r28', INPUT)]
    + list(tools.__path__)))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def manifest(task):
    return (FINE if task == 'flywire_fine' else INPUT) / 'data' / task / 'manifest.json'


def rows(task):
    return read(manifest(task))['records']


def verify():
    for root, expected in PARENTS.items():
        assert sha(root / 'protocol.json') == expected, str(root)
    protocol = read(ROOT / 'protocol.json')
    for path, expected in protocol['files'].items():
        assert sha(path) == expected, ('source drift', path)
    for task, expected in protocol['input_manifests'].items():
        assert sha(manifest(task)) == expected, task
    return protocol

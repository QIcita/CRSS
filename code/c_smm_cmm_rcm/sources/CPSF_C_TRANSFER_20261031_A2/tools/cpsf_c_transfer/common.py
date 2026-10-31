from pathlib import Path
import tools

BASE = Path('/root/megrez-tmp')
ROOT = BASE/'CPSF_C_TRANSFER_20261031_A2'
REPAIR = BASE/'CPSF_H01_PROGRESSIVE_REPAIR_20261030_A1'
TRANSFER = BASE/'CPSF_SMM_TRANSFER_20261030_A1'
FLY = BASE/'CPSF_PROGRESSIVE_SMM_20261029_A1'
FLY_DATA = BASE/'CPSF_SMM_JOINT_20261028_A1/data'
TOOLS = [str(ROOT/'tools'), str(REPAIR/'tools'), str(TRANSFER/'tools')]
tools.__path__ = list(dict.fromkeys(TOOLS + list(tools.__path__)))
from tools.cpsf_h01_progressive_repair.common import read, save, sha, state_digest, runtime, PYTHON
from tools.cpsf_smm_joint.common import CONTROLS, INPUT

TASKS = ('flywire_coarse', 'microns')


def verify():
    protocol = read(ROOT/'protocol.json')
    for path, expected in protocol['files'].items():
        assert sha(path) == expected, ('Frozen source/input drift', path)
    return protocol

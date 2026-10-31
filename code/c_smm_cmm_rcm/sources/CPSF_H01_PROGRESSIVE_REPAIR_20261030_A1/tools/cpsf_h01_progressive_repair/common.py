from pathlib import Path
import tools

BASE=Path('/root/megrez-tmp')
ROOT=BASE/'CPSF_H01_PROGRESSIVE_REPAIR_20261030_A1'
PARENT=BASE/'CPSF_SMM_TRANSFER_20261030_A1'
tools.__path__=list(dict.fromkeys([str(ROOT/'tools'),str(PARENT/'tools'),*tools.__path__]))
from tools.cpsf_smm_transfer.common import H01,INPUT,PYTHON,read,save,sha,state_digest,runtime as parent_runtime
TASKS=('h01_local','h01_global')
ARMS=('regional','regional_scaled','regional_scaled_pi')

def runtime(gpu,reserve):
    return parent_runtime(gpu,reserve)

def verify():
    p=read(ROOT/'protocol.json')
    for path,expected in p['files'].items():
        assert sha(path)==expected,('Frozen source/input drift',path)
    return p

"""Frozen progressive architecture on the existing input/training protocol."""
import hashlib
from pathlib import Path
import tools

ROOT=Path('/root/megrez-tmp/CPSF_PROGRESSIVE_SMM_20261029_A1')
RUNTIME=Path('/root/megrez-tmp/CPSF_SMM_RUNTIME_20261029_A1')
EXPERIMENT=Path('/root/megrez-tmp/CPSF_SMM_JOINT_20261028_A1')
tools.__path__=list(dict.fromkeys([str(ROOT/'tools'),str(RUNTIME/'tools'),str(EXPERIMENT/'tools'),*tools.__path__]))
from tools.cpsf_smm_runtime.common import read,save,sha,state_digest,PYTHON,runtime,CONTROLS
from tools.cpsf_smm_runtime.common import verify as verify_parent
ARMS=('progressive_gated','progressive_ungated')


def verify():
    verify_parent()
    p=read(ROOT/'protocol.json')
    assert sha(EXPERIMENT/'protocol.json')==p['parent_protocol_sha256']
    for path,expected in p['files'].items():
        assert sha(path)==expected,path
    return p

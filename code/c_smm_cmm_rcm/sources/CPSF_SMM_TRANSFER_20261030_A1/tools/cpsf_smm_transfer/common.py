"""Task adapters on frozen FlyWire cores; task-specific baseline protocols."""
import hashlib
import json
import os
import sys
from pathlib import Path

BASE=Path('/root/megrez-tmp')
ROOT=BASE/'CPSF_SMM_TRANSFER_20261030_A1'
STORE=Path('/root/CPSF_SMM_TRANSFER_20261030_A1')
PROGRESSIVE=BASE/'CPSF_PROGRESSIVE_SMM_20261029_A1'
MIC=BASE/'CPSF_MICRONS_INPUT_20261025_A1'
INPUT=BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'
H01=BASE/'CPSF_R34_FPS_REBENCH_20261021_A2'
TREE=BASE/'CPSF_GRAPHFORMER_ADAPTED_20261013'
MIC_TREE=BASE/'CPSF_BASELINE_COMPLETION_20261026_A1/microns/graphformer'
TASKS=('h01_local','h01_global','microns')
ARMS=('surface_conditioned','progressive_gated')
import tools
tools.__path__=list(dict.fromkeys([str(ROOT/'tools'),str(PROGRESSIVE/'tools'),*tools.__path__]))
from tools.cpsf_progressive_smm.common import runtime as parent_runtime, PYTHON, read,save,sha,state_digest
# Import once so the frozen SMM/CMM dependency paths are resolved before adapters.
from tools.cpsf_smm_joint.common import paths
paths()
tools.__path__=list(dict.fromkeys([str(ROOT/'tools'),str(MIC/'tools'),*tools.__path__]))


def runtime(gpu,reserve):
    env=parent_runtime(gpu,reserve)
    env.update(PYTHONDONTWRITEBYTECODE='1')
    return env


def verify():
    p=read(ROOT/'protocol.json')
    for path,expected in p['files'].items():
        assert sha(path)==expected,('Frozen source/input drift',path)
    return p

"""Execution-only successor; the experiment and its frozen sources stay intact."""
import hashlib
import sys
from pathlib import Path
import tools

RUN = Path('/root/megrez-tmp/CPSF_SMM_RUNTIME_20261029_A1')
EXPERIMENT = Path('/root/megrez-tmp/CPSF_SMM_JOINT_20261028_A1')
tools.__path__ = list(dict.fromkeys([str(RUN/'tools'), str(EXPERIMENT/'tools'), *tools.__path__]))
from tools.cpsf_smm_joint.common import read, save, sha, runtime, PYTHON, ARMS, CONTROLS
from tools.cpsf_smm_joint.common import verify as verify_experiment


def verify():
    experiment = verify_experiment()
    execution = read(RUN/'execution.json')
    assert sha(EXPERIMENT/'protocol.json') == execution['experiment_sha256']
    for path, expected in execution['files'].items():
        assert sha(path) == expected, path
    return experiment, execution


def state_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

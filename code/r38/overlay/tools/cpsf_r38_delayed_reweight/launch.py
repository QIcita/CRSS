"""Explicit one-stage launch; no automatic retry or background baseline restart."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, PYTHON, verify, read, save, sha, runtime

def main(accept):
    verify()
    stage='acceptance' if accept else 'training'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if os.readlink(p/'cwd')==str(ROOT):
                assert b'tools.cpsf_r38_delayed_reweight.worker' not in (p/'cmdline').read_bytes(),('worker already active',p.name)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    if not accept:
        gate=read(ROOT/'acceptance/r38_flywire_coarse.json')
        assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    assert not (ROOT/'results/r38_flywire_coarse/metrics.json').exists()
    assert not (ROOT/('acceptance' if accept else 'results')/'r38_flywire_coarse_failure.json').exists(), 'Review failure before retry'
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>18000,('GPU0 not sufficiently free',free)
    args=[str(PYTHON),'-u','-m','tools.cpsf_r38_delayed_reweight.worker']+(['--accept'] if accept else [])
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs'/(stage+'.log')).open('ab',buffering=0) as log:
        proc=subprocess.Popen(args,cwd=ROOT,env=runtime(0),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(stage=stage,pid=proc.pid,gpu=0,command=args,protocol_sha256=sha(ROOT/'protocol.json'),time=time.time())
    save(ROOT/'launches'/(stage+'.json'),receipt);print(receipt,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--accept',action='store_true');args=parser.parse_args();main(args.accept)

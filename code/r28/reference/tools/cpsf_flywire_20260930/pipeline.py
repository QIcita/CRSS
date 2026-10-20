"""CPU-only dependency chain; current H01/TED training is never interrupted."""
import subprocess,sys,os,fcntl
from pathlib import Path
from .build import save
ROOT=Path(__file__).resolve().parents[2]
def main():
 lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 steps=[('pilot','build',['--mode','pilot']),('contracts','contracts',[]),('full','build',['--mode','full']),('freeze','finalize',[])]
 for name,module,args in steps:
  save(ROOT/'pipeline_status.json',dict(state='running',stage=name,gpu_training_started=False))
  with (ROOT/'logs'/(name+'.log')).open('a') as log:
   subprocess.run([sys.executable,'-m','tools.cpsf_flywire_20260930.'+module,'--root',str(ROOT),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 save(ROOT/'pipeline_status.json',dict(state='data_ready',gpu_training_started=False,next='Resource profiling and configured coarse/fine classification runs; no automatic GPU jobs in dataset setup'))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e),gpu_training_started=False));raise

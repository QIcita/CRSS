import os,sys,json,time,subprocess,fcntl,shutil
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import save,sha,load
from .configure import ROOT
from .model import JOBS


def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    for task,arm in JOBS:
        m=json.loads((ROOT/'smoke'/(task+'_'+arm)/'metrics.json').read_text())
        assert m['smoke'] and m['seed']==42
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for ref in protocol['baselines'].values():
        for name,digest in ref['sha256'].items():assert sha(Path(ref['path'])/name)==digest
    # Obtain a result for both scales early; do not condition later jobs on scores.
    jobs=JOBS
    active={};done=[];failed={}
    while len(done)+len(failed)<len(jobs):
        for key,(p,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[key]
            if code==0 and (ROOT/'results'/key/'metrics.json').exists():done.append(key)
            else:failed[key]=dict(exit_code=code)
        for task,arm in jobs:
            key=task+'_'+arm
            if key in done or key in active or key in failed:continue
            if len(active)>=1:break
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
            if free<9000 or ram<10*1024**2:break
            assert shutil.disk_usage(ROOT).free>12*2**30
            out=ROOT/'results'/key
            if out.exists():
                if (out/'metrics.json').exists():done.append(key);continue
                failed[key]=dict(error='Incomplete output preserved');continue
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.3')
            log=(ROOT/'logs'/(key+'.log')).open('x')
            p=subprocess.Popen([sys.executable,'-m','tools.cpsf_v24_physical_conditioning.train','--task',task,'--arm',arm,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[key]=(p,log)
        save(ROOT/'pipeline_status.json',dict(state='running',completed=done,failed=failed,
            active={k:p.pid for k,(p,l) in active.items()},
            pending=[t+'_'+a for t,a in jobs if t+'_'+a not in done and t+'_'+a not in active and t+'_'+a not in failed]))
        if len(done)+len(failed)<len(jobs):time.sleep(15)
    assert not failed,failed
    summary={}
    for task,arm in jobs:
        reference=load(Path(protocol['baselines'][task+'_full']['path'])/'initial.pt')
        out=ROOT/'results'/(task+'_'+arm);state=load(out/'initial.pt')
        assert all(torch.equal(state[k],reference[k]) for k in reference),'Initialization mismatch'
        m=json.loads((out/'metrics.json').read_text())
        summary[task+'_'+arm]=dict(val={k:v for k,v in m['val'].items() if k!='rows'},position_gain=m['position_gain'],scale_gain=m['scale_gain'])
    save(ROOT/'summary.json',dict(new=summary,baseline={k:v['val'] for k,v in protocol['baselines'].items()}))
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={}))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)))
        save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

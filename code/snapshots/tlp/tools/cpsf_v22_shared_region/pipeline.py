import os,sys,json,time,subprocess,fcntl,shutil
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import save,sha,load
ROOT=Path(__file__).resolve().parents[2]
REPAIR=Path('/root/megrez-tmp/CPSF_H01_TOPOLOGY_REPAIR_20260930')

def call(module,args,logname):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.3')
    with (ROOT/'logs'/logname).open('x') as log:
        p=subprocess.Popen([sys.executable,'-m','tools.cpsf_v22_shared_region.'+module,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        save(ROOT/'pipeline_status.json',dict(state='preparing',stage=module,pid=p.pid));assert p.wait()==0,module+' failed'

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    for task in ('local','global'):
        m=json.loads((ROOT/'smoke'/task/'metrics.json').read_text());assert m['smoke'] and m['seed']==42
    if not (ROOT/'cache/READY.json').exists():call('prepare',[],'prepare.log')
    while not (REPAIR/'READY.json').exists():
        assert not (REPAIR/'pipeline_failure.json').exists(),'Stage 1 failed; do not train stage 2'
        save(ROOT/'pipeline_status.json',dict(state='waiting_stage1',data_ready=True));time.sleep(30)
    ready=json.loads((REPAIR/'READY.json').read_text());assert ready['complete'] and ready['point_anchor_bitwise_equal']
    assert sha(REPAIR/'comparison.json')==ready['comparison_sha256']
    save(ROOT/'stage1_dependency.json',ready)
    jobs=[(task,arm) for task in ('local','global') for arm in ('full','no_fusion','point','skeleton')]
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
            if len(active)>=2:break
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
            p=subprocess.Popen([sys.executable,'-m','tools.cpsf_v22_shared_region.train','--task',task,'--arm',arm,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[key]=(p,log)
        save(ROOT/'pipeline_status.json',dict(state='running',completed=done,failed=failed,active={k:p.pid for k,(p,l) in active.items()},pending=[t+'_'+a for t,a in jobs if t+'_'+a not in done and t+'_'+a not in active and t+'_'+a not in failed]))
        if len(done)+len(failed)<len(jobs):time.sleep(15)
    assert not failed,failed
    summary={}
    for task in ('local','global'):
        initial=None
        for arm in ('full','no_fusion','point','skeleton'):
            out=ROOT/'results'/(task+'_'+arm);state=load(out/'initial.pt')
            if initial is None:initial=state
            else:assert all(torch.equal(state[k],initial[k]) for k in state),'Initialization mismatch'
            m=json.loads((out/'metrics.json').read_text());summary[task+'_'+arm]={k:v for k,v in m['val'].items() if k!='rows'}
    save(ROOT/'summary.json',summary);save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={}))

if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

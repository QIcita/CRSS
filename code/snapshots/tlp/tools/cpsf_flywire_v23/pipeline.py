import json,os,sys,subprocess,time,fcntl,shutil
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import save,sha
from .data import ROOT
from .model import ARMS

H01=Path('/root/megrez-tmp/CPSF_V23_SCALE_ABLATION_20261001')

def resources(active_count):
    state=json.loads((H01/'pipeline_status.json').read_text())
    h01=len(state.get('active',{})) if state.get('state')=='running' else 0
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    return active_count<2 and active_count+h01<3 and free>=9000 and ram>=10*1024**2

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    gradient_check=json.loads((ROOT/'chunk_gradient_equivalence.json').read_text())
    assert gradient_check['passed'] and gradient_check['source_sha256']==sha(ROOT/'tools/cpsf_flywire_v23/model.py')
    if not (ROOT/'cache/READY.json').exists():
        save(ROOT/'pipeline_status.json',dict(state='preparing_full_cache'))
        with (ROOT/'logs/prepare.log').open('a') as log:
            subprocess.run([sys.executable,'-m','tools.cpsf_flywire_v23.prepare'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[(task,arm) for arm in ARMS for task in ('coarse','fine')]
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
            if not resources(len(active)):break
            assert shutil.disk_usage(ROOT).free>30*2**30
            out=ROOT/'results'/key
            if out.exists():
                if (out/'metrics.json').exists():done.append(key);continue
                failed[key]=dict(error='Incomplete run preserved');continue
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.25')
            log=(ROOT/'logs'/(key+'.log')).open('x')
            p=subprocess.Popen([sys.executable,'-m','tools.cpsf_flywire_v23.train','--task',task,'--arm',arm,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[key]=(p,log)
        save(ROOT/'pipeline_status.json',dict(state='training',completed=done,failed=failed,
            active={k:p.pid for k,(p,l) in active.items()},pending=[t+'_'+a for t,a in jobs if t+'_'+a not in done and t+'_'+a not in active and t+'_'+a not in failed]))
        if len(done)+len(failed)<len(jobs):time.sleep(20)
    assert not failed,failed
    checkpoints={};summary={}
    for task in ('coarse','fine'):
        initial=None
        for arm in ARMS:
            key=task+'_'+arm;out=ROOT/'results'/key;state=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
            if initial is None:initial=state
            else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
            m=json.loads((out/'metrics.json').read_text());assert not m['test_used']
            checkpoints[key]=sha(out/'final.pt');assert checkpoints[key]==m['final_sha256']
            summary[key]=m['val']
    save(ROOT/'validation_summary.json',summary)
    save(ROOT/'ALL_TRAINED.json',dict(checkpoints=checkpoints,protocol_sha256=sha(ROOT/'protocol.json'),seed=42))
    for task,arm in jobs:
        key=task+'_'+arm;dest=ROOT/'results'/key/'test/metrics.json'
        if dest.exists():continue
        while not resources(0):time.sleep(20)
        save(ROOT/'pipeline_status.json',dict(state='evaluating_test',current=key,training_complete=20))
        with (ROOT/'logs'/('test_'+key+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','tools.cpsf_flywire_v23.test','--task',task,'--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(ROOT/'test_summary.json',{t+'_'+a:json.loads((ROOT/'results'/(t+'_'+a)/'test/metrics.json').read_text())['test'] for t,a in jobs})
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={},test_evaluations=20))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

import json,os,sys,subprocess,time,fcntl,shutil
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import save,sha
from .configure import ROOT,ARMS


def resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    return free>=9000 and ram>=10*1024**2 and shutil.disk_usage(ROOT).free>20*2**30


def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest
    for arm in ARMS:
        m=json.loads((ROOT/'smoke'/('coarse_'+arm)/'metrics.json').read_text())
        assert m['smoke'] and m['epochs']==2 and m['seed']==42
    done=[];failed={};active={}
    while len(done)+len(failed)<len(ARMS):
        for key,(p,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[key]
            if code==0 and (ROOT/'results'/key/'metrics.json').exists():done.append(key)
            else:failed[key]=dict(exit_code=code)
        for arm in ARMS:
            key='coarse_'+arm
            if key in done or key in failed or key in active:continue
            if active or not resources():break
            out=ROOT/'results'/key
            if out.exists():
                if (out/'metrics.json').exists():done.append(key);continue
                failed[key]=dict(error='Incomplete run preserved');continue
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.25')
            log=(ROOT/'logs'/(key+'.log')).open('x')
            process=subprocess.Popen([sys.executable,'-m','tools.cpsf_flywire_balanced.train','--task','coarse','--arm',arm,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[key]=(process,log)
        save(ROOT/'pipeline_status.json',dict(state='training',completed=done,failed=failed,
            active={k:p.pid for k,(p,l) in active.items()},pending=['coarse_'+a for a in ARMS if 'coarse_'+a not in done and 'coarse_'+a not in active and 'coarse_'+a not in failed],test_used=False))
        if len(done)+len(failed)<len(ARMS):time.sleep(20)
    assert not failed,failed
    summary={}
    for arm in ARMS:
        out=ROOT/'results'/('coarse_'+arm);m=json.loads((out/'metrics.json').read_text())
        assert not m['smoke'] and not m['test_used'] and m['epochs']==50
        assert m['class_weights']==protocol['loss']['class_weights']
        assert m['protocol_sha256']==sha(ROOT/'protocol.json') and sha(out/'final.pt')==m['final_sha256']
        ref=protocol['reference_runs'][arm]
        for fn,digest in ref['sha256'].items():assert sha(Path(ref['path'])/fn)==digest
        a=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        b=torch.load(Path(ref['path'])/'initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,b[k]) for k,v in a.items())
        summary[arm]=dict(balanced=m['val'],original=ref['val'],
            macro_f1_delta_pp=100*(m['val']['macro_f1']-ref['val']['macro_f1']),
            accuracy_delta_pp=100*(m['val']['accuracy']-ref['val']['accuracy']))
    save(ROOT/'validation_summary.json',summary)
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={},test_used=False))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)))
        save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

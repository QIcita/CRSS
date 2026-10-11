import os,sys,json,time,subprocess,fcntl,hashlib
from pathlib import Path
import numpy as np
from tools.cpsf_v21_segment_units.io import save,sha
ROOT=Path(__file__).resolve().parents[2]
OLD=Path('/root/megrez-tmp/CPSF_V21_SEGMENT_UNITS_20260930')

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    deadline=time.time()+6*3600
    while not (ROOT/'profile_full/status.json').exists():
        assert not (ROOT/'profile_full/failure.json').exists(),'Largest-cell profile failed'
        assert time.time()<deadline
        save(ROOT/'pipeline_status.json',dict(state='waiting_profile'));time.sleep(15)
    assert json.loads((ROOT/'profile_full/status.json').read_text())['state']=='profile_complete'
    save(ROOT/'source_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')})
    done=[];results={}
    for arm in ('full','no_fusion','skeleton','point'):
        out=ROOT/'results'/arm
        if not (out/'metrics.json').exists():
            assert not out.exists(),'Incomplete existing output preserved'
            while True:
                mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                if mem>=12*1024**2 and free>=12000:break
                save(ROOT/'pipeline_status.json',dict(state='waiting_resources',arm=arm,completed=done));time.sleep(30)
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.4')
            with (ROOT/'logs'/(arm+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,'-m','tools.cpsf_h01_topology_repair.run','--checkpoint',str(OLD/'results'/('h01_'+arm)/'final.pt'),'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                save(ROOT/'pipeline_status.json',dict(state='running',arm=arm,pid=p.pid,completed=done))
                assert p.wait()==0,f'{arm} failed; preserve artifacts'
        m=json.loads((out/'metrics.json').read_text());old=json.loads((OLD/'results'/('global_'+arm)/'metrics.json').read_text())
        assert m['folds_sha256']==old['folds_sha256']
        results[arm]=dict(old={k:v['val']['mae'] for k,v in old['results'].items()},repaired={k:v['val']['mae'] for k,v in m['results'].items()})
        if arm=='point':
            a=np.load(out/'features.npz');b=np.load(OLD/'results/global_point/features.npz')
            assert np.array_equal(a['cell'],b['cell']) and np.array_equal(a['x'],b['x']),'Point anchor changed'
            assert results[arm]['old']==results[arm]['repaired'],'Point readout changed'
        done.append(arm);save(ROOT/'comparison.json',results)
    save(ROOT/'READY.json',dict(complete=True,seed=42,arms=done,point_anchor_bitwise_equal=True,comparison_sha256=sha(ROOT/'comparison.json')))
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed=[]))

if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

"""Run identity/replay gates before launching the four authorized experiments."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
ROOT=Path('/root/megrez-tmp/CPSF_RAW_POINT_ATTENTION_20261019_R28_A1')
PARENT=Path('/root/megrez-tmp/CPSF_UTONIA_KERNEL_LOCK_20261016_R9_A1')
PY='/root/megrez-tmp/CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'
sys.path.insert(0,str(ROOT))
from tools.cpsf_raw_point_attention.common import save,sha,verify

def waves(jobs):
    workers=[]
    for kind,task,rep,gpu in jobs:
        label=f'{kind}_{task}_{rep}';out=ROOT/'evidence'/f'replay_{label}.json'
        if out.exists():continue
        previous=ROOT/'evidence'/f'replay_{label}.log'
        if previous.exists():previous.rename(previous.with_suffix('.initial_helper_failure.log'))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION='.45',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        args=[PY,'-u','-m','tools.cpsf_raw_point_attention.replay','--task',task,'--kind',kind,'--out',str(out)]
        cwd=ROOT
        if kind=='parent':
            cwd=PARENT
            args=[PY,'-u','-c',f"import runpy;runpy.run_path({str(ROOT/'tools/cpsf_raw_point_attention/replay.py')!r},run_name='__main__')",'--task',task,'--kind',kind,'--out',str(out)]
        log=open(ROOT/'evidence'/f'replay_{label}.log','w');proc=subprocess.Popen(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((proc,label,log))
    for proc,label,log in workers:
        rc=proc.wait();log.close();assert rc==0,(label,rc)

def main():
    start=time.time()
    while True:
        ready=all((ROOT/'acceptance'/f'{t}.json').exists() for t in ('h01_local','h01_global'))
        if ready:break
        for t in ('h01_local','h01_global'):
            assert 'Traceback (most recent call last)' not in (ROOT/'evidence'/f'contracts_{t}.log').read_text(),t
        assert time.time()-start<3600,'Acceptance timeout'
        time.sleep(5)
    p=verify()
    for t in p['tasks']:
        c=json.loads((ROOT/'acceptance'/f'{t}.json').read_text());assert c['passed'] and c['protocol_sha256']==sha(ROOT/'protocol.json')
    waves([(kind,t,1,gpu) for t,gpu in [('h01_local',0),('h01_global',1)] for kind in ('parent','r9_control')])
    for rep in (1,2):waves([(kind,t,rep,gpu) for kind,gpu in [('r9_lpa_attention',0),('r9_lpa_uniform',1)] for t in p['tasks']])
    checks={}
    for task in p['tasks']:
        def load(kind,rep=1):return json.loads((ROOT/'evidence'/f'replay_{kind}_{task}_{rep}.json').read_text())
        parent,control=load('parent'),load('r9_control');assert parent['initial']==control['initial']
        for a,b in zip(parent['trace'],control['trace']):
            for key in ('loss','state_sha256','keys'):assert a[key]==b[key],('control drift',task,key)
        for kind in p['variants']:
            a,b=load(kind,1),load(kind,2);assert a==b,('candidate replay drift',kind,task)
            assert a['initial']==parent['initial']
            assert [x['keys'] for x in a['trace']]==[x['keys'] for x in parent['trace']]
        checks[task]=dict(parent_control_exact_two_updates=True,candidate_independent_replays_exact=True,same_initial_core_and_sample_order=True)
    save(ROOT/'contracts.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),checks=checks,acceptance_reports={t:sha(ROOT/'acceptance'/f'{t}.json') for t in p['tasks']},test_used=False))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    log=open(ROOT/'scheduler.log','a');proc=subprocess.Popen([PY,'-u','-m','tools.cpsf_raw_point_attention.scheduler'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/'launch.json',dict(scheduler_pid=proc.pid,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json'),verified_gate=True))
    print('LAUNCHED',proc.pid,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'evidence/orchestration_failure.json',dict(error=repr(e),time=time.time()));raise

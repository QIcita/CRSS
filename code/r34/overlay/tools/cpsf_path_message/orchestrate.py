"""Acceptance, independent replays and only then formal six-job launch."""
import fcntl
import subprocess
import sys
import time
from .common import *
from .model import VARIANTS


def workers(jobs):
    running=[]
    env=dict(os.environ);env.update(json.loads(Path('/root/megrez-tmp/CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json').read_text()))
    env.update(CUDA_VISIBLE_DEVICES='1',CPSF_GPU_FRACTION='.45',PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for label,args in jobs:
        log=open(ROOT/'evidence'/f'{label}.log','w')
        proc=subprocess.Popen([sys.executable,'-u','-m',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        running.append((proc,label,log))
    save(ROOT/'orchestration_status.json',dict(state='acceptance_or_replay',workers=[dict(label=l,pid=p.pid) for p,l,_ in running],time=time.time()))
    for proc,label,log in running:
        rc=proc.wait();log.close();assert rc==0,(label,rc)


def main():
    lock=open(ROOT/'orchestration.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();(ROOT/'evidence').mkdir(exist_ok=True)
    workers([(f'contracts_{t}',['tools.cpsf_path_message.contracts','--task',t]) for t in p['tasks']])
    for task in p['tasks']:
        a=json.loads((ROOT/'acceptance'/f'{task}.json').read_text());assert a['passed'] and a['protocol_sha256']==sha(ROOT/'protocol.json')
    for kind in ('parent','r28_control')+VARIANTS:
        for replica in ((1,2) if kind in VARIANTS else (1,)):
            workers([(f'replay_{kind}_{task}_{replica}',['tools.cpsf_path_message.replay','--task',task,'--kind',kind,'--out',str(ROOT/'evidence'/f'replay_{kind}_{task}_{replica}.json')]) for task in p['tasks']])
    checks={}
    for task in p['tasks']:
        read=lambda k,r=1:json.loads((ROOT/'evidence'/f'replay_{k}_{task}_{r}.json').read_text())
        old,control=read('parent'),read('r28_control');assert old['initial']==control['initial']
        for a,b in zip(old['trace'],control['trace']):
            for key in ('loss','keys','state_sha256'):assert a[key]==b[key],('R28 drift',task,key)
        for kind in VARIANTS:
            a,b=read(kind,1),read(kind,2);assert a==b,('replay drift',kind,task)
            assert a['initial']==old['initial']
            assert [x['keys'] for x in a['trace']]==[x['keys'] for x in old['trace']]
        checks[task]=dict(parent_control_exact_two_updates=True,candidate_independent_two_update_replays_exact=True,initial_core_and_sample_order_exact=True)
    save(ROOT/'contracts.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),checks=checks,
        acceptance_reports={t:sha(ROOT/'acceptance'/f'{t}.json') for t in p['tasks']},test_used=False))
    log=open(ROOT/'scheduler.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_path_message.scheduler'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/'launch.json',dict(scheduler_pid=proc.pid,protocol_sha256=sha(ROOT/'protocol.json'),time=time.time(),acceptance_and_replay_passed=True))
    save(ROOT/'orchestration_status.json',dict(state='formal_training_launched',scheduler_pid=proc.pid,time=time.time()))
    print('LAUNCHED',proc.pid,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'evidence/orchestration_failure.json',dict(error=repr(e),time=time.time()));raise

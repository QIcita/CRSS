"""Configuration experiment waits for the existing directional controller to finish."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    try:
        base=Path('/proc')/str(pid);stat=(base/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return dict(pid=pid,start_ticks=stat[19],argv=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=str((base/'cwd').resolve()))
    except (FileNotFoundError,ProcessLookupError):return None
def alive(old):
    now=identity(old['pid'])
    return now is not None and all(now[k]==old[k] for k in ['pid','start_ticks','argv','cwd'])
def resources():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    free,used,util=[int(v.strip()) for v in text.split(',')]
    ram=int(next(line.split()[1] for line in open('/proc/meminfo') if line.startswith('MemAvailable:')))/1024
    return dict(gpu_free_mib=free,gpu_used_mib=used,gpu_util_percent=util,ram_available_mib=ram)

def main(root):
    root=Path(root).resolve();runtime=root/'runtime'
    lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(root))
    from tools.cpsf_config_only.common import verify,job_key
    protocol=verify();ph=sha(root/'protocol.json')
    upstream=Path(protocol['wait_for'])
    while True:
        status=read(upstream/'pipeline_status.json')
        assert not status.get('failed') and 'failed' not in status['state'],('upstream failed',status)
        if status['state']=='complete':
            assert not status.get('active') and not status.get('pending')
            break
        assert identity(status['controller_pid']) is not None,'Upstream controller disappeared'
        save(root/'pipeline_status.json',dict(state='waiting_for_stage6',completed=[],active={},pending=protocol['jobs'],failed={},controller_pid=os.getpid(),upstream=str(upstream),unix=time.time()))
        time.sleep(15)
    gate=root/'contracts/config.json'
    if not gate.exists():
        while resources()['gpu_free_mib']<7168 or resources()['ram_available_mib']<10240:time.sleep(15)
        save(root/'pipeline_status.json',dict(state='checking_contracts',completed=[],active={},pending=protocol['jobs'],failed={},controller_pid=os.getpid(),unix=time.time()))
        with (root/'logs/contracts.log').open('a') as out:
            subprocess.run([sys.executable,'-m','tools.cpsf_config_only.contracts'],cwd=root,stdout=out,stderr=subprocess.STDOUT,check=True,
                env=dict(os.environ,CPSF_GPU_FRACTION='.30',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    c=read(gate);assert c['passed'] and c['protocol_sha256']==ph
    key=lambda j:job_key(j['task'],j['arm'],j['config'])
    jobs={key(j):j for j in protocol['jobs']}
    active={};own={};failed={}
    assert not (runtime/'controller_failure.json').exists(),'Inspect prior controller failure before resuming'
    previous=read(runtime/'active_processes.json') if (runtime/'active_processes.json').exists() else {}
    for k,v in previous.items():
        if alive(v):active[k]=v
        else:assert (root/'results'/k/'metrics.json').exists(),('prior job vanished',k)
    pending=[k for k in jobs if k not in active and not (root/'results'/k/'metrics.json').exists()]
    last_launch=time.time() if active else 0.;gate_passed=True
    log=open(runtime/'scheduler_events.jsonl','a',buffering=1)
    def event(kind,**kw):log.write(json.dumps(dict(unix=time.time(),kind=kind,**kw))+'\n')
    event('started',controller_pid=os.getpid(),protocol_sha256=ph)
    while active or pending:
        cfg=read(runtime/'scheduler_config.json');assert 2<=cfg['max_parallel']<=6
        for k,v in list(active.items()):
            proc=own.get(k);running=proc.poll() is None if proc else alive(v)
            if running:continue
            del active[k];rc=proc.returncode if proc else None
            f=root/'results'/k/'metrics.json'
            if f.exists() and rc in (0,None):
                m=read(f);assert m['identity']['protocol_sha256']==ph and not m['test_used'] and not m['smoke']
                event('completed',job=k)
            else:failed[k]=dict(exit_code=rc,reason='job ended without successful metrics');event('failed',job=k)
        complete=[k for k in jobs if (root/'results'/k/'metrics.json').exists()]
        resource=resources();now=time.time()
        limit=cfg['max_parallel'] if gate_passed else 2
        if not failed and len(active)<limit and now-last_launch>=cfg['startup_spacing_seconds']:
            candidates=list(pending)
            if candidates and resource['gpu_free_mib']>=cfg['gpu_job_allowance_mib']+cfg['gpu_headroom_mib'] and resource['ram_available_mib']>=cfg['ram_job_allowance_mib']+cfg['ram_headroom_mib']:
                assert shutil.disk_usage(root).free>15*2**30,'Disk headroom under15GiB'
                k=candidates[0];j=jobs[k]
                with (root/'logs'/(k+'.log')).open('a') as out:
                    proc=subprocess.Popen([sys.executable,'-m','tools.cpsf_config_only.train','--task',j['task'],'--arm',j['arm'],'--config',j['config']],cwd=root,
                        stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.30'))
                ident=identity(proc.pid);assert ident is not None
                active[k]=ident;own[k]=proc;pending.remove(k);last_launch=now
                event('launched',job=k,pid=proc.pid,resource=resource)
        progress={k:read(root/'results'/k/'status.json') for k in active if (root/'results'/k/'status.json').exists()}
        save(runtime/'active_processes.json',active)
        save(root/'pipeline_status.json',dict(state='draining_after_failure' if failed else 'running',phase='configuration_comparison',
            controller_pid=os.getpid(),max_parallel=limit,completed=complete,active={k:v['pid'] for k,v in active.items()},pending=[jobs[k] for k in pending],
            failed=failed,progress=progress,resource=resource,protocol_sha256=ph,test_used=False,contracts_passed=gate_passed,unix=now))
        if failed and not active:break
        if active or pending:time.sleep(15)
    assert not failed,failed
    assert gate_passed
    with (root/'logs/analyze.log').open('a') as out:
        subprocess.run([sys.executable,'-m','tools.cpsf_config_only.analyze'],cwd=root,stdout=out,stderr=subprocess.STDOUT,check=True)
    save(root/'pipeline_status.json',dict(state='complete',completed=list(jobs),active={},pending=[],failed={},contracts_passed=True,
        protocol_sha256=ph,test_used=False,unix=time.time()))
    event('all_complete',jobs=len(jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:main(args.root)
    except BaseException as exc:
        failure=dict(error=repr(exc),pid=os.getpid(),unix=time.time())
        save(args.root/'runtime/controller_failure.json',failure)
        path=args.root/'pipeline_status.json';state=read(path) if path.exists() else {}
        state.update(state='controller_failed',controller_error=failure,unix=time.time());save(path,state)
        raise

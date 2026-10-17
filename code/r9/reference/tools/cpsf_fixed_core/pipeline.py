import fcntl,json,os,subprocess,sys,time,shutil
from .common import *

def spawn(module,args,log):
    path=ROOT/'logs'/log;path.parent.mkdir(exist_ok=True)
    with path.open('x') as fp:
        return subprocess.Popen([sys.executable,'-m','tools.cpsf_fixed_core.'+module,*args],cwd=ROOT,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.38'),
            stdout=fp,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
def resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))
    return free>=10000 and ram>=8*1024**2 and shutil.disk_usage(ROOT).free>=30*2**30
def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=verify()
    assert json.loads((ROOT/'reference_contracts.json').read_text())['passed']
    done=[];failed={};active={};pending=sorted(p['jobs'],key=lambda j:(ARMS.index(j['arm']),TASKS.index(j['task'])))
    while pending or active:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            del active[key]
            if code:failed[key]=code
            else:done.append(key)
        if not failed:
            while len(active)<2 and pending and resources():
                j=pending.pop(0);key=j['task']+'_'+j['arm'];active[key]=spawn('job',['--task',j['task'],'--arm',j['arm']],key+'.log')
        save(ROOT/'pipeline_status.json',dict(state='draining_after_failure' if failed else 'running',completed=done,failed=failed,
            active={k:v.pid for k,v in active.items()},pending=pending,test_used=False))
        if failed and not active:break
        if pending or active:time.sleep(20)
    assert not failed,failed
    results={}
    for j in p['jobs']:
        key=j['task']+'_'+j['arm'];out=ROOT/'results'/key;m=json.loads((out/'metrics.json').read_text())
        assert m['protocol_sha256']==sha(ROOT/'protocol.json') and m['final_sha256']==sha(out/'final.pt')
        assert not m['smoke'] and not m['test_used']
        ref=Path(p['references'][j['task']+'/full']['path'])
        order=lambda path:[h['sample_order_sha256'] for h in json.loads(path.read_text())]
        assert order(out/'history.json')==order(ref/'history.json')
        results[key]=dict(identity=m['identity'],val=m['val'],final_sha256=m['final_sha256'])
    save(ROOT/'training_summary.json',dict(results=results,matched_reference_order=True,test_used=False))
    for module in ('efficiency','analyze'):
        proc=spawn(module,[],module+'.log');save(ROOT/'pipeline_status.json',dict(state='analyzing',stage=module,active_pid=proc.pid,completed=done,test_used=False));assert proc.wait()==0,module
    save(ROOT/'summary.json',dict(results=results,references=p['references'],analysis_sha256=sha(ROOT/'analysis.json'),efficiency_sha256=sha(ROOT/'efficiency.json'),test_used=False))
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={},test_used=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

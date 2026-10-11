import fcntl,subprocess,sys,time,shutil
from .common import *

def spawn(module,args,key):
    path=ROOT/'logs'/(key+'.log');path.parent.mkdir(exist_ok=True)
    with path.open('a') as f:return subprocess.Popen([sys.executable,'-m','tools.cpsf_fixed_nredd.'+module,*args],cwd=ROOT,
        stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.42'))
def resource():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))
    return free,ram
def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=verify()
    assert all(json.loads((ROOT/'contracts'/(t+'.json')).read_text())['passed'] for t in TASKS)
    prep=[(t,m) for m in ('joint','point','skeleton','thumbnail','uniform_budget') for t in TASKS if not(t=='ted' and m in ('thumbnail','uniform_budget'))]
    prep=[(t,m) for t,m in prep if not (ROOT/'data'/t/m/'manifest.json').exists()]
    pending=[j for j in p['jobs'] if not (ROOT/'results'/(j['task']+'_'+j['arm'])/'metrics.json').exists()]
    active={};cpus={};failed={}
    while prep or cpus or pending or active:
        for kind,processes in [('prepare',cpus),('train',active)]:
            for key,proc in list(processes.items()):
                rc=proc.poll()
                if rc is not None:
                    del processes[key]
                    if rc:failed[kind+'/'+key]=rc
        if not failed:
            _,ram=resource()
            while prep and len(cpus)<2 and ram>10*1024**2:
                t,m=prep.pop(0);key=t+'_'+m;cpus[key]=spawn('data',['--task',t,'--mode',m],'prepare_'+key);ram-=5*1024**2
            free,ram=resource()
            for j in list(pending):
                if len(active)>=2 or free<10000 or ram<8*1024**2:break
                t,a=j['task'],j['arm']
                if not (ROOT/'data'/t/modal(a)/'manifest.json').exists():continue
                if shutil.disk_usage(ROOT).free<15*2**30:failed['disk']='less than 15GiB';break
                key=t+'_'+a;active[key]=spawn('job',['--task',t,'--arm',a],key);pending.remove(j);free-=10000;ram-=4*1024**2
        completed=[j['task']+'_'+j['arm'] for j in p['jobs'] if (ROOT/'results'/(j['task']+'_'+j['arm'])/'metrics.json').exists()]
        save(ROOT/'pipeline_status.json',dict(state='draining_after_failure' if failed else 'running',completed=completed,failed=failed,active={k:v.pid for k,v in active.items()},
            preparing={k:v.pid for k,v in cpus.items()},pending=pending,preparation_pending=prep,test_used=False))
        if failed and not active and not cpus:break
        if prep or cpus or pending or active:time.sleep(20)
    assert not failed,failed
    results={}
    for t in TASKS:
        full=ROOT/'results'/(t+'_full');baseline=json.loads((full/'identity.json').read_text())
        orders=lambda d:[r['sample_order_sha256'] for r in json.loads((d/'history.json').read_text())]
        for j in (j for j in p['jobs'] if j['task']==t):
            key=t+'_'+j['arm'];d=ROOT/'results'/key;m=json.loads((d/'metrics.json').read_text())
            assert m['identity']['protocol_sha256']==sha(ROOT/'protocol.json') and m['identity']['initial_state_sha256']==baseline['initial_state_sha256']
            assert m['final_sha256']==sha(d/'final.pt') and not m['smoke'] and not m['test_used']
            assert orders(d)==orders(full)
            results[key]=m
    proc=spawn('analyze',[],'analyze');assert proc.wait()==0
    save(ROOT/'summary.json',dict(results=results,matched_order=True,test_used=False,analysis_sha256=sha(ROOT/'analysis.json')))
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=list(results),failed={},active={},pending=[],test_used=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));raise

import fcntl,json,os,subprocess,sys,time,shutil
from .common import *

def spawn(module,args,log):
    path=ROOT/'logs'/log;path.parent.mkdir(exist_ok=True)
    with path.open('x') as fp:
        return subprocess.Popen([sys.executable,'-m','tools.cpsf_v26_complete.'+module,*args],cwd=ROOT,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.38'),
            stdout=fp,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)

def resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))
    return free>=10000 and ram>=8*1024**2 and shutil.disk_usage(ROOT).free>=30*2**30

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=verify()
    prep=spawn('prepare',['flywire'],'prepare_flywire.log')
    done=[];failed={};active={};pending=[j for j in p['jobs'] if j['role']=='new_training']
    priority={'fixed_residual':0,'full':1,'mean_only':2,'no_fusion':3,'point':4,'skeleton':5,'gate_global':6}
    pending.sort(key=lambda j:(priority.get(j['arm'],7),TASKS.index(j['task'])))
    while pending or active or prep.poll() is None:
        pc=prep.poll()
        if pc is not None and pc!=0:failed['prepare_flywire']=pc
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is None:continue
            del active[key]
            if code:failed[key]=code
            else:done.append(key)
        if failed:
            # Preserve and let already-running jobs finish; never kill other experiments.
            if not active and prep.poll() is not None:break
        else:
            while len(active)<2 and resources():
                eligible=[]
                for j in pending:
                    if not (ROOT/'data'/j['task']/'READY.json').exists():continue
                    # H01 fixed-residual reproduction must pass before new-model arms.
                    if j['task'].startswith('h01') and j['arm']!='fixed_residual' and j['task']+'_fixed_residual' not in done:continue
                    eligible.append(j)
                if not eligible:break
                j=eligible[0];pending.remove(j);key=j['task']+'_'+j['arm']
                active[key]=spawn('job',['--task',j['task'],'--arm',j['arm']],key+'.log')
        save(ROOT/'pipeline_status.json',dict(state='draining_after_failure' if failed else 'running',
            completed=done,failed=failed,active={k:v.pid for k,v in active.items()},
            pending=pending,prepare_pid=prep.pid,prepare_running=prep.poll() is None,
            reused_references=[j for j in p['jobs'] if j['role']=='reused_reference'],test_used=False))
        if failed and not active and prep.poll() is not None:break
        if not pending and not active and prep.poll() is not None:break
        time.sleep(20)
    assert not failed,failed
    summary={};orders={}
    for j in p['jobs']:
        key=j['task']+'_'+j['arm']
        if j['role']=='reused_reference':
            ref=p['references'][j['task']+'/'+j['arm']];path=Path(ref['path'])
            for fn,h in ref['hashes'].items():assert sha(path/fn)==h
            summary[key]=dict(role='frozen V23 CE reference, not a V26 new training result',source=str(path),val=ref['val']);continue
        out=ROOT/'results'/key;m=json.loads((out/'metrics.json').read_text())
        assert m['protocol_sha256']==sha(ROOT/'protocol.json') and m['final_sha256']==sha(out/'final.pt')
        assert not m['smoke'] and not m['test_used']
        order=[h['sample_order_sha256'] for h in json.loads((out/'history.json').read_text())]
        if j['task'] in orders:assert orders[j['task']]==order
        else:orders[j['task']]=order
        if j['task']=='flywire_coarse':
            old=json.loads((CE/'results/coarse_full/history.json').read_text());assert order==[h['sample_order_sha256'] for h in old]
        summary[key]=dict(role='new training',identity=m['identity'],val=m['val'],final_sha256=m['final_sha256'])
    save(ROOT/'summary.json',dict(results=summary,matched_order=True,test_used=False))
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,failed={},reused=2,test_used=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

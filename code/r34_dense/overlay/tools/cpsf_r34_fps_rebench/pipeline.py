"""Finite R34 extension queue, preserving both existing benchmark schedulers."""
import fcntl,os,shutil,subprocess,time,traceback
from .common import ROOT,PARENT,FINE,PYTHON,TASKS,read,save,sha,manifest,runtime

def live(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:return False

def gpu_free():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (line.split(',') for line in out.splitlines())}

def initial_need(model,task):
    if model in ('r28','r34'):return 7000 if task=='h01_local' else 14000 if model=='r34' and task=='h01_global' else 20000 if task=='h01_global' else 10000
    if model in ('sgformer','polynormer','graphformer','tednet','tlpnet'):return 4500
    return 18500 if model in ('utonia','ptv3') else 12000

def need(root,key,model,task):
    a=root/'acceptance'/(key+'.json')
    return max(2200,int(read(a)['peak_gpu_bytes']/2**20*1.25)+1200) if a.exists() else initial_need(model,task)

def parent_free():
    free=gpu_free()
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    used={int(a):int(b) for a,b in (line.split(',') for line in out.splitlines())}
    for root in (PARENT,FINE):
        state=read(root/'campaign_status.json');jobs={j['id']:j for j in read(root/'protocol.json')['jobs']}
        active=state.get('active') or {}
        entries={active['id']:active} if 'pid' in active else active
        for key,value in entries.items():
            if not live(value['pid']):continue
            if value['kind']=='build':reserve=1500
            else:
                job=jobs[key];reserve=need(root,key,job['model'],job['task'])
            free[value['gpu']]-=max(0,reserve-used.get(value['pid'],0))
    return free

def spawn(task,kind,gpu):
    module='prepare' if kind=='prepare' else 'worker'
    args=['--task',task]+([] if kind=='prepare' else ['--model','r34']+(['--accept'] if kind=='accept' else []))
    log=ROOT/'logs'/('r34_'+task+'_'+kind+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:p=subprocess.Popen([str(PYTHON),'-u','-m','tools.cpsf_r34_fps_rebench.'+module,*args],
        cwd=ROOT,env=runtime(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,task=task,id='r34_'+task,kind=kind,gpu=gpu,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json'))
    save(ROOT/'launches'/(record['id']+'_'+kind+'.json'),record)
    return p,record

def main():
    lock=open(ROOT/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Refuse to duplicate orphaned workers after an interrupted scheduler.
    active=None;process=None
    if (ROOT/'campaign_status.json').exists():
        old=read(ROOT/'campaign_status.json').get('active')
        if old and live(old['pid']):
            assert os.readlink('/proc/'+str(old['pid'])+'/cwd')==str(ROOT)
            active=old
    started=time.time()
    from .compare import main as compare
    while True:
        if active:
            code=process.poll() if process else None if live(active['pid']) else 0
            if code is not None:
                save(ROOT/'exits'/(active['id']+'_'+active['kind']+'.json'),dict(exit_code=code,pid=active['pid'],time=time.time()))
                expected=(ROOT/'input_audits'/(active['task']+'.json') if active['kind']=='prepare' else
                    ROOT/'acceptance'/(active['id']+'.json') if active['kind']=='accept' else ROOT/'results'/active['id']/'metrics.json')
                if code or not expected.exists():save(ROOT/'failures'/(active['id']+'.json'),dict(stage=active['kind'],exit_code=code,missing_result=not expected.exists()))
                active=None;process=None
        free=parent_free()
        ram=next(int(s.split()[1])*1024 for s in open('/proc/meminfo') if s.startswith('MemAvailable:'))
        if active is None and shutil.disk_usage(ROOT).free>16*2**30 and ram>10*2**30:
            for task in TASKS:
                key='r34_'+task
                if (ROOT/'results'/key/'metrics.json').exists() or (ROOT/'failures'/(key+'.json')).exists():continue
                if not manifest(task).exists():continue
                if not (ROOT/'input_audits'/(task+'.json')).exists():
                    process,active=spawn(task,'prepare',max(free,key=free.get));break
                accepted=(ROOT/'acceptance'/(key+'.json')).exists()
                budget=need(ROOT,key,'r34',task)
                choices=[g for g,f in free.items() if f>budget+1500]
                if not choices:continue
                gpu=min(choices,key=free.get);process,active=spawn(task,'train' if accepted else 'accept',gpu);break
        completed=[t for t in TASKS if (ROOT/'results'/('r34_'+t)/'metrics.json').exists()]
        failures=[p.name for p in (ROOT/'failures').glob('*.json')]
        decision=compare()
        state='complete' if decision['audited'] else 'waiting_reference_results' if len(completed)==6 else 'running'
        save(ROOT/'campaign_status.json',dict(state=state,jobs_total=6,completed=completed,failures=failures,active=active,
             prepared=[t for t in TASKS if (ROOT/'input_audits'/(t+'.json')).exists()],
             waiting_geometry=[t for t in TASKS if not manifest(t).exists()],free_gpu_mib=gpu_free(),
             effective_free_after_parent_reserves=free,free_disk_bytes=shutil.disk_usage(ROOT).free,seconds=time.time()-started,
             current_reference=decision['reference'],concurrency='one additional R34 worker; original two schedulers unchanged'))
        if state=='complete':return
        if len(completed)+len(failures)==6 and not active:return
        time.sleep(15)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

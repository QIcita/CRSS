"""Finite build -> per-job acceptance -> training -> final evaluation queue."""
import fcntl,json,os,subprocess,time,shutil
from .common import ROOT,PYTHON,TASKS,read,save,runtime,sha

def free_gpus():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (line.split(',') for line in s.splitlines())}
def memory_available():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
def accept_need(name,task):
    if name=='r28':return 7000 if task=='h01_local' else 20000 if task=='h01_global' else 10000
    if name in ('sgformer','polynormer','graphformer','tednet','tlpnet'):return 4500
    return 18500 if name in ('utonia','ptv3') else 12000
def spawn(module,args,gpu,key):
    log=ROOT/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:p=subprocess.Popen([str(PYTHON),'-u','-m',module,*args],cwd=ROOT,env=runtime(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/'launches'/(key+'.json'),dict(pid=p.pid,gpu=gpu,command=[module,*args],time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
    return p
def main():
    lock=open(ROOT/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(ROOT/'protocol.json');active={};started=time.time()
    # Two builders overlap CPU extraction and GPU FPS within explicit reserves.
    while True:
        for key,job in list(active.items()):
            code=job['process'].poll()
            if code is not None:
                save(ROOT/'exits'/(key+'.json'),dict(exit_code=code,time=time.time(),pid=job['process'].pid))
                del active[key]
        free=free_gpus();launched=[]
        used={}
        s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        for line in s.splitlines():
            a,b=line.split(',');used[int(a)]=int(b)
        for j in active.values():free[j['gpu']]-=max(0,j['reserved']-used.get(j['process'].pid,0))
        if shutil.disk_usage(ROOT).free>14*2**30 and memory_available()>8*2**30:
            if sum(j['kind']=='build' for j in active.values())<2:
                for task in TASKS:
                    if 'build_'+task in active:continue
                    if (ROOT/'data'/task/'manifest.json').exists() or (ROOT/'preparation'/f'{task}_failure.json').exists():continue
                    gpu=max(free,key=free.get)
                    if free[gpu]>2000:
                        key='build_'+task;process=spawn('tools.cpsf_geometry_rebench.prepare',['--task',task],gpu,key)
                        active[key]=dict(process=process,kind='build',gpu=gpu,reserved=1500);free[gpu]-=1500;launched.append(key)
                    break
            for job in p['jobs']:
                key=job['id'];name=job['model'];task=job['task']
                if key in active or not (ROOT/'data'/task/'manifest.json').exists():continue
                if (ROOT/'results'/key/'metrics.json').exists():continue
                if (ROOT/'acceptance'/(key+'_failure.json')).exists() or (ROOT/'results'/(key+'_failure.json')).exists():continue
                if len([v for v in active.values() if v['kind']!='build'])>=3:break
                accepted=(ROOT/'acceptance'/(key+'.json')).exists()
                need=max(2200,int(read(ROOT/'acceptance'/(key+'.json'))['peak_gpu_bytes']/2**20*1.25)+1200) if accepted else accept_need(name,task)
                candidates=[g for g,f in free.items() if f>=need+1000]
                if not candidates:continue
                gpu=min(candidates,key=lambda g:free[g]);args=['--model',name,'--task',task]+([] if accepted else ['--accept'])
                process=spawn('tools.cpsf_geometry_rebench.worker',args,gpu,key+('_train' if accepted else '_accept'))
                active[key]=dict(process=process,kind='train' if accepted else 'accept',gpu=gpu,reserved=need)
                free[gpu]-=need;launched.append(key)
        completed=[j['id'] for j in p['jobs'] if (ROOT/'results'/j['id']/'metrics.json').exists()]
        failed=[str(f.relative_to(ROOT)) for d in ('preparation','acceptance','results') for f in (ROOT/d).glob('*_failure.json')]
        status=dict(state='complete' if len(completed)==len(p['jobs']) else 'running',jobs_total=len(p['jobs']),completed=completed,failures=failed,
            active={k:dict(pid=v['process'].pid,kind=v['kind'],gpu=v['gpu']) for k,v in active.items()},
            prepared=[t for t in TASKS if (ROOT/'data'/t/'manifest.json').exists()],free_gpu_mib=free_gpus(),
            free_disk_bytes=shutil.disk_usage(ROOT).free,seconds=time.time()-started)
        save(ROOT/'campaign_status.json',status)
        if len(completed)==len(p['jobs']):return
        time.sleep(15)

if __name__=='__main__':main()

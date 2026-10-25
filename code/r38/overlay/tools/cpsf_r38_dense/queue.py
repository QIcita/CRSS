"""Finite two-task queue: acceptance first, then memory-admitted training."""
import fcntl,json,os,subprocess,time
from .common import ROOT,BASE,PYTHON,TASKS,read,save,verify,R38

def env(gpu,fraction):
    x=dict(os.environ);x.update(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    x.update(CUDA_VISIBLE_DEVICES=str(gpu),CPSF_R38_DENSE_ROOT=str(ROOT),CPSF_CMM_PAIR_ROOT=str(R38),
       CPSF_R34_FPS_ROOT=str(BASE/'CPSF_R34_FPS_REBENCH_20261021_A2'),
       CPSF_REBENCH_ROOT=str(BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3'),
       PYTHONPATH=str(BASE/'CPSF_GEOMETRY_FPS_REBENCH_20261020_A3/deps'),
       CPSF_GPU_FRACTION=str(fraction),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return x

def resources():
    cmd=['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits']
    rr=[]
    for line in subprocess.check_output(cmd,text=True).splitlines():
        i,uuid,total,free=[v.strip() for v in line.split(',')];rr.append(dict(gpu=int(i),uuid=uuid,total=float(total),free=float(free)))
    usage={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        pid,used=[v.strip() for v in line.split(',')]
        try:usage[int(pid)]=float(used)
        except ValueError:pass
    return rr,usage

def launch(task,gpu,phase,reservation,total):
    folder=ROOT/'logs';folder.mkdir(exist_ok=True)
    log=open(folder/(task+'_'+phase+'.log'),'ab',buffering=0)
    args=[str(PYTHON),'-u','-m','tools.cpsf_r38_dense.worker','--task',task]
    if phase=='acceptance':args+=['--accept']
    process=subprocess.Popen(args,cwd=ROOT,env=env(gpu,min(.90,reservation/total)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(process=process,task=task,gpu=gpu,phase=phase,reservation_mib=reservation,pid=process.pid)

def public(active):return [{k:v for k,v in job.items() if k!='process'} for job in active]

def main():
    lock=open(ROOT/'queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    active=[];failed=[];complete=[]
    # Both extreme-shape acceptance suites finish before training admission.
    for task in TASKS:
        if (ROOT/'acceptance'/('r38_'+task+'.json')).exists():continue
        while True:
            gpu,_=resources();eligible=[g for g in gpu if g['free']>=10000]
            if eligible:break
            save(ROOT/'queue_status.json',dict(phase='waiting_acceptance_memory',task=task,resources=gpu,pid=os.getpid()))
            time.sleep(20)
        chosen=max(eligible,key=lambda g:g['free']);job=launch(task,chosen['gpu'],'acceptance',10000,chosen['total'])
        while job['process'].poll() is None:
            save(ROOT/'queue_status.json',dict(phase='acceptance',active=public([job]),pid=os.getpid()))
            time.sleep(5)
        path=ROOT/'acceptance'/('r38_'+task+'.json')
        if job['process'].returncode or not path.exists():
            save(ROOT/'queue_status.json',dict(phase='failed_acceptance',task=task,returncode=job['process'].returncode,pid=os.getpid()))
            return
    pending=list(TASKS)
    while pending or active:
        for job in active[:]:
            if job['process'].poll() is None:continue
            path=ROOT/'results'/('r38_'+job['task'])/'metrics.json'
            if job['process'].returncode==0 and path.exists():complete.append(job['task'])
            else:failed.append(dict(task=job['task'],returncode=job['process'].returncode))
            active.remove(job)
        gpu,usage=resources()
        for task in pending[:]:
            if (ROOT/'results'/('r38_'+task)/'metrics.json').exists():complete.append(task);pending.remove(task);continue
            peak=read(ROOT/'acceptance'/('r38_'+task+'.json'))['peak_gpu_bytes']/2**20
            need=max(2400,peak*1.12+768)
            eligible=[]
            for g in gpu:
                outstanding=sum(max(0,j['reservation_mib']-usage.get(j['pid'],0)) for j in active if j['gpu']==g['gpu'])
                if g['free']-outstanding>=need+256:eligible.append((g['free']-outstanding,g))
            if not eligible:continue
            _,chosen=max(eligible,key=lambda x:x[0]);job=launch(task,chosen['gpu'],'training',need,chosen['total'])
            active.append(job);pending.remove(task)
        save(ROOT/'queue_status.json',dict(phase='training' if active else 'waiting_training_memory',pending=pending,
                active=public(active),complete=complete,failed=failed,resources=gpu,pid=os.getpid()))
        if pending or active:time.sleep(10)
    save(ROOT/'queue_status.json',dict(phase='complete' if not failed else 'finished_with_failure',complete=complete,failed=failed,pid=os.getpid()))

if __name__=='__main__':main()

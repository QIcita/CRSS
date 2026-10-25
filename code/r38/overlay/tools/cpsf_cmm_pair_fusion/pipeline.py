"""Finite two-task queue with original workers' memory reservations retained."""
import fcntl,os,shutil,subprocess,time,traceback
from .common import ROOT,BASE,PARENT,GEOMETRY,FINE,PYTHON,TASKS,MODEL,read,save,sha,runtime,add_backend
add_backend()
from tools.cpsf_r34_fps_rebench.pipeline import gpu_free,live,need,parent_free


def available(active):
    free=parent_free()
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    used={int(a):int(b) for a,b in (s.split(',') for s in out.splitlines())}
    old=read(PARENT/'campaign_status.json').get('active')
    if old and live(old['pid']):
        reserve=need(PARENT,old['id'],'r34',old['task'])
        free[old['gpu']]-=max(0,reserve-used.get(old['pid'],0))
    ap=BASE/'CPSF_R34_ADAPTIVE_PATCH10_20261022_A1/queue_status.json'
    if ap.exists():
        for entry in read(ap).get('active',[]):
            if live(entry['pid']):
                free[entry['gpu']]-=max(0,entry['reservation_mib']-used.get(entry['pid'],0))
    # Additional headroom for CB2; no other process is changed.
    for pid in used:
        try:
            cwd=os.readlink('/proc/'+str(pid)+'/cwd')
            if 'MORPHWEAVE_CB2_' not in cwd:continue
            env=open('/proc/'+str(pid)+'/environ','rb').read().split(bytes([0]))
            visible=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            if visible.isdigit() and int(visible) in free:free[int(visible)]-=512
        except (OSError,StopIteration):pass
    for key,a in active.items():
        free[a['gpu']]-=max(0,a['reserve_mib']-used.get(a['pid'],0))
    return free


def main():
    lock=open(ROOT/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};children={};began=time.time()
    if (ROOT/'campaign_status.json').exists():
        for key,a in read(ROOT/'campaign_status.json').get('active',{}).items():
            if live(a['pid']):
                assert os.readlink('/proc/'+str(a['pid'])+'/cwd')==str(ROOT);active[key]=a
    while True:
        for key,a in list(active.items()):
            code=children[key].poll() if key in children else None if live(a['pid']) else 0
            if code is None:continue
            result=ROOT/'acceptance'/(key+'.json') if a['kind']=='accept' else ROOT/'results'/key/'metrics.json'
            save(ROOT/'exits'/(key+'_'+a['kind']+'.json'),dict(pid=a['pid'],exit_code=code,time=time.time()))
            if code or not result.exists():save(ROOT/'failures'/(key+'.json'),dict(stage=a['kind'],exit_code=code,result_missing=not result.exists()))
            del active[key];children.pop(key,None)
        free=available(active)
        ram=next(int(s.split()[1])*1024 for s in open('/proc/meminfo') if s.startswith('MemAvailable:'))
        if shutil.disk_usage(ROOT).free>12*2**30 and ram>10*2**30:
            for task in TASKS:
                key=MODEL+'_'+task
                if key in active or (ROOT/'results'/key/'metrics.json').exists() or (ROOT/'failures'/(key+'.json')).exists():continue
                accepted=(ROOT/'acceptance'/(key+'.json')).exists()
                reserve=(max(1200,int(read(ROOT/'acceptance'/(key+'.json'))['peak_gpu_bytes']/2**20*1.08)+600)
                    if accepted else 1200 if task=='h01_local' else 9000)
                choices=[g for g,v in free.items() if v>reserve+350]
                if not choices:continue
                gpu=min(choices,key=free.get);kind='train' if accepted else 'accept'
                log=ROOT/'logs'/(key+'_'+kind+'.log');log.parent.mkdir(exist_ok=True)
                args=[str(PYTHON),'-u','-m','tools.cpsf_cmm_pair_fusion.worker','--model',MODEL,'--task',task]
                if not accepted:args.append('--accept')
                env=runtime(gpu,task)
                if kind=='accept' and task=='h01_global':env['CPSF_GPU_FRACTION']='.36'
                with log.open('a') as f:p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                a=dict(pid=p.pid,task=task,id=key,kind=kind,gpu=gpu,reserve_mib=reserve,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json'))
                active[key]=a;children[key]=p;save(ROOT/'launches'/(key+'_'+kind+'.json'),a)
                free[gpu]-=reserve
        completed=[t for t in TASKS if (ROOT/'results'/(MODEL+'_'+t)/'metrics.json').exists()]
        failures=[p.name for p in (ROOT/'failures').glob('*.json')]
        save(ROOT/'campaign_status.json',dict(state='complete' if len(completed)==2 else 'running',completed=completed,failures=failures,
            active=active,jobs_total=2,free_gpu_mib=gpu_free(),effective_free_mib=free,disk_free_bytes=shutil.disk_usage(ROOT).free,seconds=time.time()-began))
        if not active and len(completed)+len(failures)==2:
            from .review import main as review
            review();return
        time.sleep(15)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

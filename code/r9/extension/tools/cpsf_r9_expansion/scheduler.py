"""Fixed three-task queue; reclaims no unrelated processes."""
import os,sys,time,subprocess,fcntl,json
from tools.cpsf_utonia_context.common import ROOT,save,sha,verify

def main():
    lock=open(ROOT/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    jobs=[dict(task='flywire_coarse',gpu=0,fraction='.85'),dict(task='ted',gpu=1,fraction='.28'),dict(task='tlp',gpu=1,fraction='.65')]
    active=[]
    for j in jobs:
        out=ROOT/'results'/('patch_rms_010_'+j['task']);j['out']=str(out)
        if (out/'metrics.json').exists():j['state']='complete';continue
        log=open(ROOT/'logs'/(j['task']+'.log'),'a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(j['gpu']),CPSF_GPU_FRACTION=j['fraction'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        proc=subprocess.Popen([sys.executable,'-m','tools.cpsf_r9_expansion.train','--task',j['task']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        j.update(state='running',pid=proc.pid,started=time.time());active.append((j,proc,log))
    while True:
        for j,p,log in active:
            code=p.poll()
            if code is not None and j['state']=='running':
                j.update(state='complete' if code==0 else 'failed',exit_code=code);log.close()
        save(ROOT/'queue_status.json',dict(jobs=jobs,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
        if all(j['state']!='running' for j in jobs):break
        time.sleep(10)
if __name__=='__main__':main()

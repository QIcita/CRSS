import fcntl,subprocess,sys,time
from .common import *

def main():
    lock=open(ROOT/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    q=dict(state='running',protocol_sha256=sha(ROOT/'protocol.json'),jobs=[]);workers=[]
    for task,gpu in [('flywire_coarse',0),('ted',1),('tlp',1)]:
        d=ROOT/'results'/('r28_lpa_attention_'+task);d.mkdir(parents=True,exist_ok=True)
        assert not (d/'latest.pt').exists() and not (d/'metrics.json').exists(),'Do not repeat formal jobs'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION=('.40' if task=='flywire_coarse' else '.18' if task=='ted' else '.33'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=open(d/'worker.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_r28_expansion.train','--task',task],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        j=dict(task=task,kind='r28_lpa_attention',gpu=gpu,pid=proc.pid,state='running');q['jobs'].append(j);workers.append((proc,j,log))
    while True:
        for proc,j,log in workers:
            rc=proc.poll()
            if rc is not None:j.update(state='complete' if rc==0 else 'failed',returncode=rc)
        q['time']=time.time()
        if all(proc.poll() is not None for proc,_,_ in workers):q['state']='complete' if all(j['state']=='complete' for _,j,_ in workers) else 'failed'
        save(ROOT/'queue_status.json',q)
        if q['state']!='running':break
        time.sleep(15)
    for _,_,log in workers:log.close()
if __name__=='__main__':main()

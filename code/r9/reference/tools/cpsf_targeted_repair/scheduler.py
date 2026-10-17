"""Four matched jobs, two per GPU, with immutable acceptance gate."""
import subprocess,time,os
from .common import *
def main():
    verify();gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    running=[]
    for task in ['ted','tlp']:
        for gpu,arm in enumerate(['control','repair']):
            folder=ROOT/'results'/f'{task}_{arm}';folder.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION='.40',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            log=open(folder/'worker.log','a')
            command=[str(BASE/'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/env/bin/python'),'-u','-m','tools.cpsf_targeted_repair.train','--task',task,'--arm',arm]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running.append(dict(task=task,arm=arm,gpu=gpu,pid=process.pid,process=process,log=log))
    while True:
        rows=[]
        for job in running:
            code=job['process'].poll();folder=ROOT/'results'/f"{job['task']}_{job['arm']}"
            metrics=folder/'metrics.json'
            rows.append({k:v for k,v in job.items() if k not in ('process','log')}|dict(exit_code=code,state='running' if code is None else ('complete' if code==0 and metrics.exists() else 'failed')))
        save(ROOT/'queue_status.json',dict(jobs=rows,protocol_sha256=sha(ROOT/'protocol.json'),timestamp=time.time()))
        if all(j['exit_code'] is not None for j in rows):break
        time.sleep(30)
if __name__=='__main__':main()

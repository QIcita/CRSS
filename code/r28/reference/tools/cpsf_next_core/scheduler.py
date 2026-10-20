import subprocess,time,os,sys,shutil
from .common import *
def main():
 verify();gate=json.loads((ROOT/'contracts.json').read_text());assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json');assert shutil.disk_usage(ROOT).free>4*2**30
 running=[]
 for task,gpu in [('h01_local',0),('h01_global',1),('ted',0)]:
  out=ROOT/'results'/f'{task}_full';out.mkdir(parents=True,exist_ok=True);log=open(out/'worker.log','a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION=('.75' if task=='h01_global' else '.40'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
  proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_next_core.train','--task',task],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);running.append(dict(task=task,gpu=gpu,pid=proc.pid,process=proc))
 while True:
  jobs=[]
  for job in running:
   code=job['process'].poll();out=ROOT/'results'/f"{job['task']}_full";jobs.append({k:v for k,v in job.items() if k!='process'}|dict(exit_code=code,state='running' if code is None else 'complete' if code==0 and (out/'metrics.json').exists() else 'failed'))
  save(ROOT/'queue_status.json',dict(jobs=jobs,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
  if all(j['exit_code'] is not None for j in jobs):break
  time.sleep(30)
if __name__=='__main__':main()

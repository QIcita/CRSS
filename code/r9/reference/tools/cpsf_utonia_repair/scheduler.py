import os,sys,time,subprocess,shutil
from .common import *
def main():
 verify();c=json.loads((ROOT/'contracts.json').read_text());assert c['passed'] and c['protocol_sha256']==sha(ROOT/'protocol.json');assert shutil.disk_usage(ROOT).free>4*2**30
 # Four paired H01 tests; existing TED jobs on GPU0 remain untouched.
 jobs=[]
 for kind,task,gpu in [('order_control','h01_global',1),('bounded_patch','h01_global',1),('order_control','h01_local',1),('bounded_patch','h01_local',1)]:
  out=ROOT/'results'/f'{kind}_{task}';out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION='.40' if task=='h01_global' else '.08',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');log=open(out/'worker.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_utonia_repair.train','--task',task,'--kind',kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(kind=kind,task=task,gpu=gpu,pid=proc.pid,process=proc))
 while True:
  current=[]
  for j in jobs:
   code=j['process'].poll();out=ROOT/'results'/f"{j['kind']}_{j['task']}";current.append({k:v for k,v in j.items() if k!='process'}|dict(exit_code=code,state='running' if code is None else 'complete' if code==0 and (out/'metrics.json').exists() else 'failed'))
  save(ROOT/'queue_status.json',dict(jobs=current,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
  if all(x['exit_code'] is not None for x in current):break
  time.sleep(30)
if __name__=='__main__':main()

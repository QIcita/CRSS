"""Only H01; first-epoch replay gate before amplitude jobs; final replay audit."""
import os,sys,time,subprocess,fcntl,shutil
from .common import *

def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

def main():
 lock=open(ROOT/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=verify();contracts=json.loads((ROOT/'contracts.json').read_text());assert contracts['passed'] and contracts['protocol_sha256']==sha(ROOT/'protocol.json')
 assert shutil.disk_usage(ROOT).free>8*2**30
 jobs=[]
 for task in p['tasks']:
  for kind,suffix,gpu in [('patch_rms_010','',0),('patch_rms_010','_replay',1),('patch_rms_030','',0 if task=='h01_local' else 1)]:
   out=ROOT/'results'/(f'{kind}_{task}'+suffix)
   assert not out.exists(),('preexisting run',str(out))
   jobs.append(dict(task=task,kind=kind,suffix=suffix,gpu=gpu,out=str(out),state='pending'))
 gate=None
 while True:
  for j in jobs:
   if j['state']=='running' and j['proc'].poll() is not None:
    j['exit_code']=j['proc'].returncode;j['log'].close();j['state']='complete' if j['exit_code']==0 and (Path(j['out'])/'metrics.json').exists() else 'failed'
  controls=[j for j in jobs if j['kind']=='patch_rms_010']
  if gate is None and all((Path(j['out'])/'epoch1.pt').exists() for j in controls):
   checks={}
   for task in p['tasks']:
    pair=[j for j in controls if j['task']==task]
    a,b=[torch.load(Path(j['out'])/'epoch1.pt',map_location='cpu',weights_only=False) for j in pair]
    checks[task]={k:exact(a[k],b[k]) for k in ['model','optimizer','scheduler','steps','torch_rng','cuda_rng','numpy_rng']}
    checks[task]['loss']=a['history'][0]['loss']==b['history'][0]['loss']
    checks[task]['order']=a['history'][0]['sample_order_sha256']==b['history'][0]['sample_order_sha256']
   gate=all(all(d.values()) for d in checks.values());save(ROOT/'replay_epoch1_gate.json',dict(passed=gate,checks=checks,protocol_sha256=sha(ROOT/'protocol.json')))
  if any(j['state']=='failed' for j in controls):gate=False
  for j in jobs:
   if j['state']!='pending':continue
   if j['kind']=='patch_rms_030':
    if gate is False:j['state']='blocked_replay_failed';continue
    if gate is None:continue
   free=int(subprocess.check_output(['nvidia-smi',f'--id={j["gpu"]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free<(8500 if j['task']=='h01_global' else 2200):continue
   out=Path(j['out']);out.mkdir(parents=True)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(j['gpu']),CPSF_REPLICA_SUFFIX=j['suffix'],CPSF_GPU_FRACTION='.40' if j['task']=='h01_global' else '.12',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   log=open(out/'worker.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_utonia_context.train','--task',j['task'],'--kind',j['kind']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   j.update(state='running',pid=proc.pid,proc=proc,log=log,started=time.time())
  save(ROOT/'queue_status.json',dict(jobs=[{k:v for k,v in j.items() if k not in ['proc','log']} for j in jobs],replay_epoch1_passed=gate,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
  if all(j['state'] in ['complete','failed','blocked_replay_failed'] for j in jobs):break
  time.sleep(10)
 final={}
 for task in p['tasks']:
  pair=[j for j in controls if j['task']==task]
  if not all(j['state']=='complete' for j in pair):final[task]={'passed':False,'reason':'control incomplete'};continue
  a,b=[torch.load(Path(j['out'])/'latest.pt',map_location='cpu',weights_only=False) for j in pair]
  checks={k:exact(a[k],b[k]) for k in ['model','optimizer','scheduler','steps','torch_rng','cuda_rng','numpy_rng']}
  x,y=[np.load(Path(j['out'])/'val_predictions.npz') for j in pair];checks['predictions']=all(np.array_equal(x[k],y[k]) for k in x.files)
  final[task]=dict(passed=all(checks.values()),checks=checks)
 save(ROOT/'replay_final_gate.json',dict(passed=all(x['passed'] for x in final.values()),tasks=final))

if __name__=='__main__':main()

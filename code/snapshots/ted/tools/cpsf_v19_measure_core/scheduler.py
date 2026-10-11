"""Durable dependency scheduler. No background monitoring service required."""
import subprocess,time,os,sys,fcntl
from .io import *
ROOT=Path(__file__).resolve().parents[2];PY=sys.executable

def main():
 lock=open(ROOT/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((ROOT/'contracts_final.json').read_text())['cuda']['passed']
 assert (ROOT/'cache_h01/manifest.json').is_file()
 for name in ('h01_measure','h01_point','ted_measure'):
  assert (ROOT/'smoke_final'/name/'metrics.json').is_file(),name
 assert (ROOT/'smoke_final/global_measure/manifest.json').is_file()
 assert (ROOT/'smoke_final/global_point/manifest.json').is_file()
 jobs=[]
 # Prioritize H01 paired evidence and dependent global readouts before the long TED matrix.
 for arm in ('measure','point','skeleton','count','geometry','wrong'):
  jobs.append(dict(name='h01_'+arm,kind='h01',module='train',args=['--task','h01','--arm',arm,'--cache',str(ROOT/'cache_h01')],out=ROOT/'results'/('h01_'+arm),dependencies=[]))
 for arm in ('measure','point','skeleton','count','geometry','wrong'):
  jobs.append(dict(name='global_'+arm,kind='global',module='global_task',args=['--checkpoint',str(ROOT/'results'/('h01_'+arm)/'final.pt')],out=ROOT/'results'/('global_'+arm),dependencies=['h01_'+arm]))
 for arm in ('measure','skeleton','count','wrong'):
  jobs.append(dict(name='ted_'+arm,kind='ted',module='train',args=['--task','ted','--arm',arm],out=ROOT/'results'/('ted_'+arm),dependencies=[]))
 active={};complete=set();failed={};started=time.time();source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
 save(ROOT/'launch_manifest.json',dict(seed=42,source_sha256=source,protocol_sha256=sha(ROOT/'CPSF_V19_MEASURE_CORE_PROTOCOL_20260930.md'),jobs=[dict(j,out=str(j['out'])) for j in jobs]))
 while len(complete)+len(failed)<len(jobs):
  for name,(proc,j,log) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   log.close();del active[name]
   if rc==0 and (j['out']/'metrics.json').exists():complete.add(name)
   else:failed[name]=dict(returncode=rc,out=str(j['out']))
  for j in jobs:
   name=j['name']
   if name in complete or name in failed or name in active:continue
   if any(dep in failed for dep in j['dependencies']):failed[name]=dict(blocked_by=j['dependencies']);continue
   if not all(dep in complete for dep in j['dependencies']):continue
   if len(active)>=2:break
   kinds=[x[1]['kind'] for x in active.values()]
   if j['kind'] in ('ted','global') and any(k in ('ted','global') for k in kinds):continue
   if j['kind']=='ted' and any(k.startswith('global') for k in active):continue
   available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
   if available<8*2**30:continue
   if j['out'].exists():
    # Completed outputs may be acknowledged; incomplete outputs never overwritten or resumed silently.
    if (j['out']/'metrics.json').is_file():complete.add(name);continue
    failed[name]=dict(error='Existing incomplete output; manual review required');continue
   cmd=[PY,'-m','tools.cpsf_v19_measure_core.'+j['module'],*j['args'],'--out',str(j['out'])]
   env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.42')
   log=open(ROOT/'logs'/(name+'.log'),'w');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   active[name]=(proc,j,log);print('START',name,proc.pid,flush=True)
  save(ROOT/'scheduler_status.json',dict(state='running',active={name:dict(pid=p.pid,kind=j['kind'],out=str(j['out'])) for name,(p,j,l) in active.items()},completed=sorted(complete),failed=failed,pending=[j['name'] for j in jobs if j['name'] not in complete|set(failed)|set(active)],seconds=time.time()-started))
  time.sleep(15)
 save(ROOT/'scheduler_status.json',dict(state='complete' if not failed else 'finished_with_failures',completed=sorted(complete),failed=failed,active={},pending=[],seconds=time.time()-started))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'scheduler_failure.json',dict(error=repr(e)));raise

"""Engineering gate; intentionally tiny cohorts do not estimate accuracy."""
import subprocess,sys,time,os
from .io import *
from .model import Model
ROOT=Path(__file__).resolve().parents[2]
def call(module,args):
 cmd=[sys.executable,'-m','tools.cpsf_v19_measure_core.'+module,*map(str,args)];print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
def main():
 call('contracts',['--out',ROOT/'contracts_final.json'])
 while not (ROOT/'cache_h01/manifest.json').is_file():time.sleep(10)
 setup();rows=json.loads((ROOT/'cache_h01/manifest.json').read_text())['records'];r=max(rows,key=lambda r:r['points']);item=load(ROOT/'cache_h01/local'/(r['sample']+'.pt'));g=item['geometry'];checks={}
 for arm in ('measure','point','skeleton','count','geometry','wrong'):
  setup();m=Model('h01',arm).cuda();torch.cuda.reset_peak_memory_stats();out=m([g],[item['query']]);out.square().sum().backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());checks[arm]=dict(parameters=sum(v.numel() for v in m.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,finite=True);del m,out;torch.cuda.empty_cache()
 save(ROOT/'largest_local_contract.json',dict(sample=r['sample'],points=r['points'],patches=g['m'],nodes=g['n'],arms=checks))
 for task,arm in [('h01','measure'),('h01','point'),('ted','measure')]:
  args=['--task',task,'--arm',arm,'--smoke','--out',ROOT/'smoke_final'/(task+'_'+arm)]
  if task=='h01':args+=['--cache',ROOT/'cache_h01']
  call('train',args)
 for arm in ('measure','point'):
  call('global_task',['--checkpoint',ROOT/'smoke_final'/('h01_'+arm)/'final.pt','--out',ROOT/'smoke_final'/('global_'+arm),'--smoke'])
 save(ROOT/'preflight_complete.json',dict(passed=True,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},scope='engineering only; no efficacy claim'))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(ROOT/'preflight_failure.json',dict(error=repr(e)));raise

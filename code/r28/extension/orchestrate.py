import os,sys,subprocess,time,json,tarfile,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from tools.cpsf_r28_expansion.common import save,sha,verify
PY=sys.executable

def run(module,args,label,gpu=0,rep='1'):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION='.45',CPSF_ACCEPT_REP=rep,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 log=open(R/'evidence'/(label+'.log'),'w');p=subprocess.Popen([PY,'-u','-m',module]+args,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 return p,log,label

def wait(workers):
 for p,log,label in workers:
  rc=p.wait();log.close();assert rc==0,(label,rc)

def main():
 p=verify();assert json.loads((R/'input_audit.json').read_text())['passed']
 # New shared mixin must reproduce original R28 H01 states/outputs exactly.
 code="""
from tools.cpsf_r28_expansion.common import *
from tools.cpsf_r28_expansion.model import Model
from tools.cpsf_raw_point_attention.model import Model as Old
from tools.cpsf_raw_point_attention.data import load_item,move
from torch.nn import functional as F
traces={}
for task in ('h01_local','h01_global'):
 traces[task]=[]
 for factory in (lambda:Old(task,'r9_lpa_attention'),lambda:Model(task)):
  setup();m=factory().cuda();rr=[r for r in rows(task) if r['split']=='train'];order=[rr[i] for i in np.random.permutation(len(rr))];batch=8 if task=='h01_local' else 4
  scale=max(float(np.std([r['target'] for r in rr])),.05)
  if task=='h01_global':
   mean=float(np.mean([r['target'] for r in rr]))
   with torch.no_grad():m.head.bias.fill_(float(np.log(mean/(1-mean))))
  trace=[state_sha(m)];opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
  for step in range(2):
   opt.zero_grad(set_to_none=True);losses=[]
   for r in order[step*batch:(step+1)*batch]:
    z=m(move(load_item(r)));y=torch.tensor(r['target'],device='cuda');loss=((z-y.float())/scale).square() if task=='h01_global' else F.cross_entropy(z[None],y.long()[None],label_smoothing=.05)
    (loss/batch).backward();losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();trace.append(dict(state_sha256=state_sha(m),loss=losses))
  traces[task].append(trace);del m,opt;torch.cuda.empty_cache()
 assert traces[task][0]==traces[task][1],('H01 adapter drift',task)
save(ROOT/'evidence/h01_initial_exact.json',dict(passed=True,full_two_update_traces=traces))
"""
 # Models' identical original functions are additionally compared by two-step traces.
 (R/'evidence/h01_check.py').write_text(code)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CPSF_GPU_FRACTION='.45',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with open(R/'evidence/h01_check.log','w') as log:subprocess.run([PY,'-u','-c',code],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 for task,gpu in [('flywire_coarse',0),('ted',1),('tlp',0)]:
  for rep in ('1','2'):wait([run('tools.cpsf_r28_expansion.contracts',['--task',task],task+'_accept_'+rep,gpu,rep)])
  a=json.loads((R/'acceptance'/(task+'_rep1.json')).read_text());b=json.loads((R/'acceptance'/(task+'_rep2.json')).read_text())
  assert a['passed'] and b['passed'] and a['trace']==b['trace'],('replay drift',task)
 save(R/'contracts.json',dict(passed=True,protocol_sha256=sha(R/'protocol.json'),all_input_hashes_valid=True,initial_h01_states_exact=True,independent_two_update_replays_exact=True,max_sample_backward=True,tests_used=False))
 with tarfile.open(R/'source_frozen.tar.gz','w:gz') as t:
  for name in p['source_sha256']:t.add(R/name,arcname=name)
  t.add(R/'protocol.json',arcname='protocol.json')
 save(R/'source_archive_identity.json',dict(sha256=sha(R/'source_frozen.tar.gz'),bytes=(R/'source_frozen.tar.gz').stat().st_size))
 log=open(R/'scheduler.log','a');proc=subprocess.Popen([PY,'-u','-m','tools.cpsf_r28_expansion.scheduler'],cwd=R,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(R/'launch.json',dict(scheduler_pid=proc.pid,protocol_sha256=sha(R/'protocol.json'),time=time.time(),verified_gate=True));print('LAUNCHED',proc.pid,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(R/'evidence/orchestration_failure.json',dict(error=repr(e),time=time.time()));raise

"""Finite search then selection, matched refits and global retention checks."""
import subprocess,time,sys,os,fcntl
from .io import *
from .model import ARMS
from .train import validated_protocol
ROOT=Path(__file__).resolve().parents[2]

def call(module,args):
    subprocess.run([sys.executable,'-m','tools.cpsf_v21_segment_units.'+module,*map(str,args)],cwd=ROOT,check=True)

def run_jobs(jobs,stage,parallel):
    active={};complete=set();failed={};started=time.time()
    while len(complete)+len(failed)<len(jobs):
        for name,(p,j,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[name]
            if code==0 and (j['out']/'metrics.json').exists():complete.add(name)
            else:failed[name]=dict(code=code,out=str(j['out']))
        for j in jobs:
            name=j['name']
            if name in complete or name in failed or name in active:continue
            if len(active)>=parallel:break
            ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if ram<10*2**30 or free<8192:continue
            assert __import__('shutil').disk_usage(ROOT).free>8*2**30,'Disk headroom below 8 GiB'
            if j['out'].exists():
                if (j['out']/'metrics.json').exists():complete.add(name);continue
                failed[name]=dict(error='Existing incomplete output preserved');continue
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CPSF_GPU_FRACTION='.25',PYTHONPATH=str(ROOT))
            log=open(ROOT/'logs'/(name+'.log'),'w')
            cmd=[sys.executable,'-m','tools.cpsf_v21_segment_units.'+j['module'],*j['args'],'--out',str(j['out'])]
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[name]=(p,j,log);print('START',stage,name,p.pid,flush=True)
        save(ROOT/'pipeline_status.json',dict(state='running',stage=stage,active={n:dict(pid=p.pid,out=str(j['out'])) for n,(p,j,l) in active.items()},completed=sorted(complete),failed=failed,pending=[j['name'] for j in jobs if j['name'] not in complete|set(failed)|set(active)],stage_seconds=time.time()-started))
        if len(complete)+len(failed)<len(jobs):time.sleep(15)
    save(ROOT/(stage+'_status.json'),dict(completed=sorted(complete),failed=failed))
    assert not failed,failed

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=validated_protocol()
    if not (ROOT/'contracts.json').exists():call('contracts',[])
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    if not (ROOT/'smoke/metrics.json').exists():call('train',['--phase','inner','--candidate','full','--smoke','--out',ROOT/'smoke'])
    jobs=[dict(name='inner_'+c['name'],module='train',args=['--phase','inner','--candidate',c['name']],out=ROOT/'inner'/c['name']) for c in protocol['candidates']]
    run_jobs(jobs,'inner_search',2)
    if not (ROOT/'selection.json').exists():
        scores=[]
        for i,c in enumerate(protocol['candidates']):
            path=ROOT/'inner'/c['name']/'metrics.json';m=json.loads(path.read_text());assert not m['protocol']['smoke']
            for epoch,row in m['inner_validation'].items():scores.append(dict(candidate=c,epoch=int(epoch),macro_f1=row['macro_f1'],order=i,metrics_sha256=sha(path)))
        best=max((v for v in scores if v['candidate']['name']=='full'),key=lambda z:(z['macro_f1'],-z['epoch']))
        controls={v['candidate']['name']:v['macro_f1'] for v in scores if v['epoch']==best['epoch'] and v['candidate']['name']!='full'}
        assert set(controls)=={'no_fusion','node_units'}
        passed=all(best['macro_f1']>value for value in controls.values())
        save(ROOT/'selection.json',dict(**best,all_scores=scores,controls_at_selected_epoch=controls,gate_passed=passed,search_protocol_sha256=sha(ROOT/'search_protocol.json'),selection_scope='full inner validation only; controls gate continuation; no outer score read'))
    selection=json.loads((ROOT/'selection.json').read_text())
    if not selection['gate_passed']:
        save(ROOT/'pipeline_status.json',dict(state='complete',stage='inner_gate_not_met',active={},failed={},inner_runs=3,local_refits=0,global_transfers=0,selection=selection))
        return
    jobs=[dict(name='h01_'+arm,module='train',args=['--phase','refit','--arm',arm],out=ROOT/'results'/('h01_'+arm)) for arm in ARMS]
    run_jobs(jobs,'matched_refit',2)
    jobs=[dict(name='global_'+arm,module='global_task',args=['--checkpoint',str(ROOT/'results'/('h01_'+arm)/'final.pt')],out=ROOT/'results'/('global_'+arm)) for arm in ('full','no_fusion','no_context','no_scale','point','skeleton')]
    run_jobs(jobs,'global_transfer',1)
    save(ROOT/'pipeline_status.json',dict(state='complete',stage='finished',active={},failed={},inner_runs=3,local_refits=8,global_transfers=6,selection=json.loads((ROOT/'selection.json').read_text())))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)))
        previous=json.loads((ROOT/'pipeline_status.json').read_text()) if (ROOT/'pipeline_status.json').exists() else {}
        save(ROOT/'pipeline_status.json',dict(previous,state='failed',error=repr(e)))
        raise

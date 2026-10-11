"""Build once; wait for shared GPU slot; then validate and train all arms."""
import json,os,sys,subprocess,time,fcntl,shutil,datetime
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
from .plan import ROOT,OLD,BAL,ARMS

BUILD_PY='/root/megrez-tmp/CPSF_FLYWIRE_TASKS_20260930/venv/bin/python'


def run(module,args,logname,python=None):
    logpath=ROOT/'logs'/logname
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CPSF_GPU_FRACTION='.25')
    with logpath.open('x') as log:
        p=subprocess.Popen([python or sys.executable,'-m','tools.cpsf_flywire_coarse_cohort.'+module,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        save(ROOT/'pipeline_status.json',dict(state='running',stage=module,args=args,active_pid=p.pid,test_used=False))
        code=p.wait()
    assert code==0,(module,args,code)


def resources():
    previous=json.loads((BAL/'pipeline_status.json').read_text())
    if previous['state']=='failed':raise RuntimeError('Prior balanced comparison failed; inspect before new training')
    if previous['state']!='complete':return False
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    return free>=9000 and ram>=10*1024**2 and shutil.disk_usage(ROOT).free>30*2**30


def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((ROOT/'pilot_geometry.json').read_text())['passed']
    run('acquire',[],'acquire_full.log',BUILD_PY)
    run('prepare',[],'prepare_full.log')
    while not resources():
        save(ROOT/'pipeline_status.json',dict(state='waiting_gpu_slot',reason='Wait for old balanced comparison completion and resource headroom',data_ready=True,test_used=False))
        time.sleep(30)
    run('configure',[],'configure.log')
    run('contracts',[],'contracts.log')
    for arm in ARMS:
        run('train',['--task','coarse','--arm',arm,'--out',str(ROOT/'smoke'/('coarse_'+arm)),'--smoke'],'smoke_'+arm+'.log')
    done=[]
    for arm in ARMS:
        while not resources():time.sleep(30)
        key='coarse_'+arm
        run('train',['--task','coarse','--arm',arm,'--out',str(ROOT/'results'/key)],key+'.log')
        done.append(key);save(ROOT/'training_progress.json',dict(completed=done,test_used=False))
    protocol=json.loads((ROOT/'protocol.json').read_text());summary={};orders=None
    for arm in ARMS:
        out=ROOT/'results'/('coarse_'+arm);m=json.loads((out/'metrics.json').read_text());ref=json.loads((out/'frozen_reference/metrics.json').read_text())
        assert m['total_optimizer_steps']==62200 and m['unique_train_cells_seen']==protocol['train_pool']
        assert m['protocol_sha256']==sha(ROOT/'protocol.json') and sha(out/'final.pt')==m['final_sha256']
        assert not m['test_used'] and not m['smoke']
        order=[r['sample_order_sha256'] for r in json.loads((out/'history.json').read_text())]
        if orders is None:orders=order
        else:assert order==orders
        summary[arm]=dict(expanded=m['val'],legacy=m['val_legacy'],added=m['val_added'],frozen_reference=ref,
            legacy_macro_f1_delta_pp=100*(m['val_legacy']['macro_f1']-ref['val_legacy']['macro_f1']),
            expanded_macro_f1_delta_pp=100*(m['val']['macro_f1']-ref['val']['macro_f1']))
    save(ROOT/'validation_summary.json',summary)
    save(ROOT/'pipeline_status.json',dict(state='complete',completed=done,test_used=False))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)));save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

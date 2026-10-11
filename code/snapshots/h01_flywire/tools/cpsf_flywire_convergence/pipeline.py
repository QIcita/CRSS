import datetime,fcntl,json,os,shutil,subprocess,sys,time
from .common import ROOT,COHORT,OLD,ARMS,save,sha,verify_sources

def run(module,args=(),name=None):
    log=ROOT/'logs'/((name or module)+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('x') as fp:
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CPSF_GPU_FRACTION='.25')
        proc=subprocess.Popen([sys.executable,'-m','tools.cpsf_flywire_convergence.'+module,*args],
            cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        save(ROOT/'pipeline_status.json',dict(state='running',stage=module,args=list(args),active_pid=proc.pid,test_used=False))
        code=proc.wait()
    assert code==0,(module,args,code)

def enough_resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    ram=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))
    return free>=9000 and ram>=10*1024**2 and shutil.disk_usage(ROOT).free>=30*2**30

def main():
    lock=open(ROOT/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((ROOT/'protocol.json').read_text());verify_sources(ROOT,p)
    assert json.loads((ROOT/'contracts.json').read_text())['passed']
    for arm in ARMS:
        m=json.loads((ROOT/'smoke'/('coarse_'+arm)/'metrics.json').read_text())
        assert m['smoke'] and not m['test_used'] and m['epochs']==2
        assert m['protocol_sha256']==sha(ROOT/'protocol.json')
    while True:
        states={r.name:json.loads((r/'pipeline_status.json').read_text())['state'] for r in (COHORT,OLD)}
        assert not any(s=='failed' for s in states.values()),states
        if all(s=='complete' for s in states.values()) and enough_resources():break
        save(ROOT/'pipeline_status.json',dict(state='waiting_predecessors',stage='1-finalize-existing-runs',
            predecessor_states=states,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_used=False))
        time.sleep(30)
    run('registry',name='registry_before_diagnosis')
    run('diagnose')
    assert json.loads((ROOT/'diagnostics/summary.json').read_text())['passed']
    done=[]
    for arm in ARMS:
        while not enough_resources():time.sleep(30)
        run('train',['--task','coarse','--arm',arm,'--out',str(ROOT/'results'/('coarse_'+arm))],name='coarse_'+arm)
        done.append(arm);save(ROOT/'training_progress.json',dict(completed=done,test_used=False))
    summary={};expected=None
    for arm in ARMS:
        out=ROOT/'results'/('coarse_'+arm);m=json.loads((out/'metrics.json').read_text())
        ref=p['reference_runs'][arm];weighted=json.loads((COHORT/'results'/('coarse_'+arm)/'metrics.json').read_text())
        assert sha(COHORT/'results'/('coarse_'+arm)/'metrics.json')==ref['sha256']['metrics.json']
        assert m['final_sha256']==sha(out/'final.pt') and m['protocol_sha256']==sha(ROOT/'protocol.json')
        assert not m['smoke'] and not m['test_used'] and m['total_optimizer_steps']==62200 and m['unique_train_cells_seen']==6039
        order=[h['sample_order_sha256'] for h in json.loads((out/'history.json').read_text())]
        oldorder=[h['sample_order_sha256'] for h in json.loads((COHORT/'results'/('coarse_'+arm)/'history.json').read_text())]
        assert order==oldorder,'Ordinary and weighted CE did not see the same sequence'
        if expected is None:expected=order
        else:assert order==expected
        summary[arm]=dict(ordinary_CE={k:m[k] for k in ('val','val_legacy','val_added')},
            weighted_CE={k:weighted[k] for k in ('val','val_legacy','val_added')},identity=m['identity'])
    save(ROOT/'objective_comparison.json',dict(test_used=False,matched_order=True,matched_initialization=True,
        matched_updates=62200,arms=summary,
        next_stage='Review objective tradeoffs and geometry diagnosis; freeze one objective before the separate gated-interaction experiment'))
    run('registry',name='registry_after_CE')
    save(ROOT/'pipeline_status.json',dict(state='complete',completed_stages=[1,2,3],
        pending_stages=['4-gated-interaction-after-objective-review','5-H01-and-single-modality-validation'],
        completed=done,test_used=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'pipeline_failure.json',dict(error=repr(e)))
        save(ROOT/'pipeline_status.json',dict(state='failed',error=repr(e)));raise

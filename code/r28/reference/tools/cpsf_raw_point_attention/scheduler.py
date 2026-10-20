"""Four predeclared H01 experiments, two per GPU, no adaptive model search."""
import fcntl,subprocess,sys,time
from .common import *

def audit(kind):
    from .train import calc
    p=verify();result={}
    for task,cfg in p['tasks'].items():
        d=ROOT/'results'/f'{kind}_{task}';m=json.loads((d/'metrics.json').read_text());identity=m['identity']
        assert identity['version']==ROOT.name and identity['kind']==kind and identity['task']==task
        assert identity['protocol_sha256']==sha(ROOT/'protocol.json') and identity['manifest_sha256']==sha(ROOT/'data'/task/'manifest.json')
        assert identity['seed']==42 and not identity['pretrained'] and not m['test_used']
        assert (m['epochs'],m['steps'])==(cfg['epochs'],cfg['updates']) and sha(d/'final.pt')==m['final_sha256']
        with np.load(d/'val_predictions.npz') as a:
            rr=[r for r in rows(task) if r['split']=='val'];assert a['keys'].tolist()==[r['key'] for r in rr]
            assert np.array_equal(a['target'],np.array([r['target'] for r in rr]));metrics=calc(a['target'],a['logits'],task)
        assert metrics==m['val']
        hist=json.loads((d/'history.json').read_text());old=json.loads((Path(p['comparison_reference'][task]['root'])/'history.json').read_text())
        assert len(hist)==len(old)==cfg['epochs']
        assert all((a['sample_order_sha256'],a['lr'],a['steps'])==(b['sample_order_sha256'],b['lr'],b['steps']) for a,b in zip(hist,old))
        result[task]=metrics
    local=result['h01_local']['macro_f1']-p['comparison_reference']['h01_local']['metrics']['macro_f1']
    global_gain=p['comparison_reference']['h01_global']['metrics']['mae']-result['h01_global']['mae']
    result.update(local_gain=local,global_mae_reduction=global_gain,promoted=local>=0 and global_gain>=0 and (local>=.001 or global_gain>=.0001))
    save(ROOT/f'audit_{kind}.json',result);return result

def main():
    lock=open(ROOT/'scheduler.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();gate=json.loads((ROOT/'contracts.json').read_text())
    assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=os.readlink(proc/'cwd')
        except OSError:continue
        assert not (cwd==str(ROOT) and '-m tools.cpsf_raw_point_attention.train ' in cmd),'Existing formal worker'
    jobs=[];q=dict(state='running',protocol_sha256=sha(ROOT/'protocol.json'),jobs=[],audits={})
    for kind,gpu in [('r9_lpa_attention',0),('r9_lpa_uniform',1)]:
        for task in p['tasks']:
            out=ROOT/'results'/f'{kind}_{task}';out.mkdir(parents=True,exist_ok=True)
            if (out/'metrics.json').exists():continue
            assert not (out/'failure.json').exists(),'Inspect failure before restart'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CPSF_GPU_FRACTION='.45',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            log=open(out/'worker.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_raw_point_attention.train','--task',task,'--kind',kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job=dict(kind=kind,task=task,gpu=gpu,pid=proc.pid,state='running');q['jobs'].append(job);jobs.append((proc,job,log))
    while any(proc.poll() is None for proc,_,_ in jobs):
        for proc,job,_ in jobs:
            code=proc.poll()
            if code is not None:job.update(returncode=code,state='complete' if code==0 else 'failed')
        q['time']=time.time();save(ROOT/'queue_status.json',q);time.sleep(15)
    for proc,job,log in jobs:log.close();job.update(returncode=proc.returncode,state='complete' if proc.returncode==0 else 'failed')
    if any(proc.returncode!=0 for proc,_,_ in jobs):q['state']='failed'
    else:
        for kind in p['variants']:q['audits'][kind]=audit(kind)
        q['state']='complete'
    save(ROOT/'queue_status.json',q)
if __name__=='__main__':main()

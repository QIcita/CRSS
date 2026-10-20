"""Bounded resource-aware six-job queue, leaving all existing runs untouched."""
import fcntl
import subprocess
import sys
import time
from .common import *
from .model import VARIANTS


def audit(kind):
    from .train import calc
    p=verify();result={}
    for task,cfg in p['tasks'].items():
        out=ROOT/'results'/f'{kind}_{task}';m=json.loads((out/'metrics.json').read_text());ident=m['identity']
        assert ident['version']==ROOT.name and ident['kind']==kind and ident['task']==task
        assert ident['protocol_sha256']==sha(ROOT/'protocol.json') and ident['manifest_sha256']==sha(ROOT/'data'/task/'manifest.json')
        assert ident['seed']==42 and not ident['pretrained'] and not m['test_used']
        assert (m['epochs'],m['steps'])==(cfg['epochs'],cfg['updates']) and sha(out/'final.pt')==m['final_sha256']
        with np.load(out/'val_predictions.npz') as a:
            rr=[r for r in rows(task) if r['split']=='val']
            assert a['keys'].tolist()==[r['key'] for r in rr] and np.array_equal(a['target'],np.array([r['target'] for r in rr]))
            values=calc(a['target'],a['logits'],task)
        assert values==m['val']
        hist=json.loads((out/'history.json').read_text());old=json.loads((Path(p['comparison_reference'][task]['root'])/'history.json').read_text())
        assert len(hist)==len(old)==cfg['epochs']
        assert all((a['sample_order_sha256'],a['lr'],a['steps'])==(b['sample_order_sha256'],b['lr'],b['steps']) for a,b in zip(hist,old))
        result[task]=values
    local=result['h01_local']['macro_f1']-p['comparison_reference']['h01_local']['metrics']['macro_f1']
    global_gain=p['comparison_reference']['h01_global']['metrics']['mae']-result['h01_global']['mae']
    result.update(local_gain=local,global_mae_reduction=global_gain,promoted=local>=0 and global_gain>=0 and (local>=.001 or global_gain>=.0001))
    save(ROOT/f'audit_{kind}.json',result);return result


def main():
    lock=open(ROOT/'scheduler.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();gate=json.loads((ROOT/'contracts.json').read_text())
    assert gate['passed'] and gate['protocol_sha256']==sha(ROOT/'protocol.json')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=os.readlink(proc/'cwd')
        except OSError:continue
        assert not (cwd==str(ROOT) and '-m tools.cpsf_path_message.train ' in cmd),'Existing formal worker'
    pending=[(kind,task) for kind in VARIANTS for task in p['tasks'] if not (ROOT/'results'/f'{kind}_{task}'/'metrics.json').exists()]
    running={};done=[];failed=[];audits={}
    env=dict(os.environ);env.update(json.loads(Path('/root/megrez-tmp/CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json').read_text()))
    env.update(CUDA_VISIBLE_DEVICES='1',CPSF_GPU_FRACTION='.45',PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    while pending or running:
        for name,job in list(running.items()):
            rc=job['process'].poll()
            if rc is not None:
                job['log'].close();record=dict(kind=job['kind'],task=job['task'],gpu=1,pid=job['process'].pid,returncode=rc)
                (done if rc==0 and (ROOT/'results'/name/'metrics.json').exists() else failed).append(record);del running[name]
        if pending and not failed and len(running)<6:
            free={int(a):int(b) for a,b in (line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines())}
            txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
            usage={int(a):int(b) for a,b in (line.split(',') for line in txt.splitlines())} if txt else {}
            kind,task=pending[0];name=f'{kind}_{task}'
            report=json.loads((ROOT/'acceptance'/f'{task}.json').read_text())['variants'][kind]
            need=int(max(c['peak_gpu_bytes'] for c in report['cases'])/2**20*1.25+850)
            reserve=sum(max(0,j['reserve_mib']-usage.get(j['process'].pid,0)) for j in running.values())
            if free.get(1,0)-reserve>need+2000:
                out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
                assert not (out/'failure.json').exists(),'Inspect failure before resume'
                log=open(out/'worker.log','a');proc=subprocess.Popen([sys.executable,'-u','-m','tools.cpsf_path_message.train','--task',task,'--kind',kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[name]=dict(process=proc,log=log,kind=kind,task=task,reserve_mib=need);pending.pop(0)
                save(out/'launch.json',dict(pid=proc.pid,gpu=1,time=time.time(),protocol_sha256=sha(ROOT/'protocol.json')))
        for kind in VARIANTS:
            if kind not in audits and all((ROOT/'results'/f'{kind}_{t}'/'metrics.json').exists() for t in p['tasks']):
                audits[kind]=audit(kind)
        save(ROOT/'queue_status.json',dict(state='failed' if failed else 'running' if running or pending else 'complete',
            protocol_sha256=sha(ROOT/'protocol.json'),pending=[dict(kind=k,task=t) for k,t in pending],
            running={n:dict(pid=j['process'].pid,gpu=1,kind=j['kind'],task=j['task'],reserve_mib=j['reserve_mib']) for n,j in running.items()},
            completed=done,failed=failed,audits=audits,time=time.time(),stages_2_and_3_enabled=False))
        if failed and not running:break
        if running or pending:time.sleep(5)


if __name__=='__main__':main()

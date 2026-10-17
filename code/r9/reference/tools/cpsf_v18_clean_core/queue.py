"""Fail-closed staged screen, seed42. Never edits or interrupts old workers.

Verification jobs are separate from the 14 new scientific endpoints. The
three interventions finish all four tasks in order; ready tasks within an
intervention run concurrently subject to global memory/worker admission.
"""
import argparse,collections,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path

BASE=Path('/root/megrez-tmp')
HOME=BASE/'CPSF_V15_STAGED_20260924'
OLD=BASE/'CPSF_V14_TRIRELATION_20260922'
PRIOR=BASE/'CPSF_SEED42_SCREENING_20260923'
PYTHON=BASE/'H01_CPSF_PILOT_20260919/.venv/bin/python'
MODULE='tools.cpsf_v15_train_20260924.'
HELPER=BASE/'CPSF_PARALLEL_SCHEDULER_20260922/cpsf_parallel_scheduler_20260922.py'
spec=importlib.util.spec_from_file_location('resource_helper',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read,save,sha=helper.read,helper.save,helper.sha
CORE=('encoder.py','geometry.py','offload.py','decoders.py','common.py','train.py','cell.py','test_contracts.py')


def make_jobs(root):
    jobs=[]
    def append(key,mod,args,expected,deps=(),kind='smoke',stage=None,task=None,variant='dynamic',endpoint=False):
        args=list(map(str,args));out=Path(args[args.index('--out')+1])
        command=[str(PYTHON),'-u','-m',MODULE+mod,*args]
        if mod=='audit':command=[str(PYTHON),'-u',str(OLD/'tools/tlp_missing_edge_ap_report.py'),*args]
        jobs.append(dict(id=key,command=command,out=str(out),expected=str(expected),deps=list(deps),kind=kind,
            stage=stage,task=task,variant=variant,endpoint=endpoint,state='pending'))
        return key
    pre=[]
    # Full dataset tensors stay in OLD/cache_h01; no new joint sample cache.
    for stage in ('residual','routing','role'):
        for task in ('h01','ted','tlp'):
            out=root/'smoke'/stage/task
            args=['--stage',stage,'--task',task,'--variant','dynamic','--seed',42,'--smoke','--out',out]
            if task=='h01':args+=['--cache',OLD/'smoke_cache_v1']
            pre.append(append(f'smoke/{stage}/{task}','train',args,out/'metrics.json',stage=stage,task=task))
        out=root/'verification'/f'largest_local_{stage}.json'
        pre.append(append(f'stress/local/{stage}','stress',['--stage',stage,'--kind','local','--cache',OLD/'cache_h01','--out',out],
            out,kind='stress_local',stage=stage))
    out=root/'verification/largest_cell_role.json'
    pre.append(append('stress/cell/role','stress',['--stage','role','--kind','cell','--out',out],out,kind='stress_cell',stage='role'))
    small=HOME/'cache_point_smoke'
    cache=HOME/'cache_point_full'
    key=append('point/cache_smoke','point_cache',['--cache',OLD/'smoke_cache_v1','--out',small,'--smoke'],small/'manifest.json',kind='cache')
    pre.append(key)
    pointsmoke=root/'smoke/point/h01'
    key=append('smoke/point/h01','train',['--stage','role','--task','h01','--variant','point','--seed',42,'--smoke','--cache',small,'--out',pointsmoke],
        pointsmoke/'metrics.json',[key],task='h01',stage='role',variant='point');pre.append(key)
    for variant,checkpoint,dependency in [('dynamic',root/'smoke/role/h01/final.pt','smoke/role/h01'),('point',pointsmoke/'final.pt',key)]:
        feat=root/'smoke'/variant/'cell/features';out=root/'smoke'/variant/'cell/readout'
        k=append(f'smoke/{variant}/cell_extract','cell',['--stage','extract','--seed',42,'--smoke','--checkpoint',checkpoint,'--out',feat],
            feat/'metrics.json',[dependency],kind='smoke_extract',variant=variant,stage='role')
        k=append(f'smoke/{variant}/cell_fit','cell',['--stage','fit','--seed',42,'--smoke','--features',feat,'--out',out],
            out/'metrics.json',[k],kind='smoke_fit',variant=variant,stage='role');pre.append(k)
    out=root/'verification/largest_point_cell.json'
    pre.append(append('stress/cell/point','stress',['--stage','role','--kind','point_cell','--out',out],out,kind='stress_cell',stage='role'))
    pointcache=append('point/cache_full','point_cache',['--cache',OLD/'cache_h01','--out',cache],cache/'manifest.json',
        ['point/cache_smoke'],kind='cache')
    pre.append(pointcache)
    previous=list(pre)
    for stage in ('residual','routing','role'):
        endpoints=[]
        for task in ('h01','ted','tlp'):
            out=root/'results'/stage/task/'42/dynamic'
            args=['--stage',stage,'--task',task,'--variant','dynamic','--seed',42,'--out',out]
            if task=='h01':args+=['--cache',OLD/'cache_h01']
            k=append(f'{stage}/{task}/train','train',args,out/'metrics.json',previous,'train',stage,task,endpoint=True)
            endpoints.append(k)
            if task=='tlp':
                audit=out/'missing_ap_independent.json'
                endpoints.append(append(f'{stage}/tlp/audit','audit',['--run',f'dynamic={out}','--out',audit],audit,[k],'audit',stage,'tlp'))
        feat=root/'results'/stage/'cell/42/dynamic/features';out=feat.parent/'readout'
        k=append(f'{stage}/cell/extract','cell',['--stage','extract','--seed',42,'--checkpoint',root/'results'/stage/'h01/42/dynamic/final.pt','--out',feat],
            feat/'metrics.json',[f'{stage}/h01/train'],'extract',stage,'cell')
        k=append(f'{stage}/cell/fit','cell',['--stage','fit','--seed',42,'--features',feat,'--out',out],out/'metrics.json',[k],'fit',stage,'cell',endpoint=True)
        endpoints.append(k);previous=endpoints
    # H01-only point control. A single surface-only core suffices for all three
    # interventions: cross branches have zero incidences and no skeleton nodes.
    out=root/'results/point/h01/42/point'
    k=append('point/h01/train','train',['--stage','role','--task','h01','--variant','point','--seed',42,'--cache',cache,'--out',out],
        out/'metrics.json',pre,'train','role','h01','point',True)
    feat=root/'results/point/cell/42/point/features';out=feat.parent/'readout'
    k=append('point/cell/extract','cell',['--stage','extract','--seed',42,'--checkpoint',root/'results/point/h01/42/point/final.pt','--out',feat],
        feat/'metrics.json',[k],'extract','role','cell','point')
    append('point/cell/fit','cell',['--stage','fit','--seed',42,'--features',feat,'--out',out],out/'metrics.json',[k],'fit','role','cell','point',True)
    assert len({j['id'] for j in jobs})==len(jobs)
    ids={j['id'] for j in jobs}
    assert all(set(j['deps'])<=ids for j in jobs)
    assert sum(j['endpoint'] for j in jobs)==14
    return jobs


def freeze(sources):
    for p,digest in sources.items():assert sha(p)==digest,'Frozen source drift: '+p


def collect_sources():
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for p in (HELPER,HOME/'PROTOCOL_ZH.md',OLD/'cache_h01/manifest.json',OLD/'smoke_cache_v1/manifest.json',
              OLD/'tools/tlp_missing_edge_ap_report.py',BASE/'H01_HIGH_COVERAGE_FULL104_20260920/manifest.json',
              BASE/'CPSF_SHARED_V4_20260919/cache_ted/construction_report.json'):
        sources[str(p)]=sha(p)
    for p in (OLD/'tools/cpsf_v14_train_20260922').glob('*.py'):sources[str(p)]=sha(p)
    for device in ('cpu','cuda'):
        path=HOME/'verification'/f'contracts_{device}_final.json';d=read(path);assert d['status']=='passed'
        for name in CORE:assert d['source_sha256'][name]==sha(Path(__file__).parent/name),'Outdated gate: '+name
        sources[str(path)]=sha(path)
    path=HOME/'verification/queue_final.json';d=read(path);assert d['passed']
    for name,digest in d['source_sha256'].items():assert digest==sha(Path(__file__).parent/name)
    sources[str(path)]=sha(path)
    # Inherit data-loader and official metric dependencies frozen by V14.
    old_sources=read(PRIOR/'plan.json')['sources']
    for path,digest in old_sources.items():
        if '/NREDD_proj/NREDD/' in path:sources[path]=digest
    freeze(sources)
    return sources


def snapshot_baselines(root):
    rows=[]
    for task in ('h01','ted','tlp','cell'):
        for variant in ('dynamic','skeleton','fixed','once'):
            folder=PRIOR/'results/v14'/task/'42'/variant
            metric=folder/('readout/metrics.json' if task=='cell' else 'metrics.json')
            if not metric.exists():continue
            d=read(metric);q=d['protocol']
            assert q['seed']==42
            audit=folder/'missing_ap_independent.json'
            audited=task!='tlp' or (audit.exists() and read(audit)['runs'][variant]['historical_metrics_reproduced'])
            rows.append(dict(task=task,variant=variant,path=str(metric),sha256=sha(metric),audited=audited,
                metrics=d,checkpoint=str(metric.parent/'final.pt')))
    save(root/'v14_baselines.json',dict(rows=rows,scope='Read-only V14 controls; official NREDD comparison remains a separate protocol/source'))


def profile(j):
    kind=j['kind']
    if kind in ('cache','audit','fit','smoke_fit'):return dict(gpu=0,ram=3072,heavy=False)
    if kind in ('stress_cell','extract'):return dict(gpu=2048,ram=14336,heavy=True)
    if kind=='smoke_extract':return dict(gpu=2048,ram=4096,heavy=True)
    if kind=='smoke':return dict(gpu=3072,ram=3072,heavy=False)
    if kind=='stress_local':return dict(gpu=8192,ram=6144,heavy=False)
    return dict(gpu=6144 if j['task']=='h01' else 8192,ram=6144 if j['task']=='h01' else 7168,heavy=False)


def ram_growth(profile,actual):
    """Bound reservation to observed growth, retaining a type-specific cap.

    Full nominal RAM is reserved for a new worker.  For already-running
    workers, reserving the entire unused nominal budget serialized otherwise
    safe jobs even when the host had substantial available memory.  The cap
    remains conservative and is added to the measured RSS.
    """
    cap=6144 if profile['heavy'] else 3072
    return min(max(0,profile['ram']-actual),cap)


def admission(j,r,active):
    # Admit alongside prior workers using their reserved growth budgets.
    # No prior worker is cancelled, paused or reprioritized by this queue.
    need=profile(j);reserve_gpu=reserve_ram=0.;gpu_workers=set(r['gpu_by_pid']);heavy=0
    managed={x['identity']['pid'] for x in active}
    for x in active:
        pid=x['identity']['pid'];p=profile(x)
        reserve_gpu+=max(0,p['gpu']-r['gpu_by_pid'].get(pid,0))
        reserve_ram+=ram_growth(p,r['processes'].get(pid,{}).get('rss_mib',0))
        heavy+=p['heavy']
        if p['gpu']:gpu_workers.add(pid)
    for pid,p in r['processes'].items():
        if pid in managed:continue
        cmd=p['cmd'];worker=('cpsf_' in cmd and ('--task ' in cmd or '--stage extract' in cmd))
        if not worker and pid not in r['gpu_by_pid']:continue
        isheavy='--stage extract' in cmd;heavy+=isheavy;gpu_workers.add(pid)
        budget=1536 if '--variant skeleton' in cmd else 2048 if isheavy else 8192
        reserve_gpu+=max(0,budget-r['gpu_by_pid'].get(pid,0))
        reserve_ram+=ram_growth(dict(ram=14336 if isheavy else 7168,heavy=isheavy),p['rss_mib'])
    if len(active)>=3:return False,'managed_slots'
    if need['gpu'] and len(gpu_workers)>=4:return False,'global_gpu_slots'
    if need['heavy'] and heavy:return False,'heavy_extractor_limit'
    if r['gpu_free_mib']-reserve_gpu-need['gpu']<3072:return False,'gpu_budget'
    if r['ram_available_mib']-reserve_ram-need['ram']<4096:return False,'ram_budget'
    if shutil.disk_usage(HOME).free<8*2**30:return False,'disk_reserve'
    return True,'admitted'


def validate(j):
    d=read(j['expected']);out=Path(j['out']);kind=j['kind']
    assert not (out/'failure.json').exists(),'Worker failure evidence exists'
    if kind in ('stress_cell','stress_local'):
        assert d['passed'] and d['no_subsampling']
        for name in CORE:assert d['source_sha256'][name]==sha(Path(__file__).parent/name)
    elif kind=='cache':
        assert len(d['records'])==(6 if 'smoke' in j['id'] else 930)
        assert d['point_only_encoding'] and d['physical_scale_um']==1
    elif kind=='audit':
        assert not d['pending'] and d['runs'][j['variant']]['historical_metrics_reproduced']
        parent=read(Path(j['out']).parent/'metrics.json')['test']['missing_edge_macro_ap']
        assert 0<=parent<=1
        assert abs(d['runs'][j['variant']]['metrics']['missing']['macro_ap']-parent)<1e-10,'Independent missing-AP mismatch'
    elif kind in ('extract','smoke_extract'):
        assert d['passed'];m=read(out/'manifest.json')
        assert len(m['records'])==(2 if kind=='smoke_extract' else 104)
        assert m['protocol']['seed']==42 and m['protocol']['variant']==j['variant'] and m['protocol']['stage']==j['stage']
        for row in m['records']:assert sha(out/(row['cell']+'.pt'))==row['sha256']
    else:
        q=d['protocol'];assert q['seed']==42 and q['variant']==j['variant'] and q['stage']==j['stage']
        smoke=kind.startswith('smoke');assert q['smoke_only']==smoke
        if kind in ('fit','smoke_fit'):assert q['epochs']==(2 if smoke else 200)
        else:
            assert q['task']==j['task'] and q['epochs']==(2 if smoke else {'h01':35,'ted':30,'tlp':80}[j['task']])
            for name in CORE:assert q['source_sha256'][name]==sha(Path(__file__).parent/name)
            if not smoke:
                baseline=read(PRIOR/'results/v14'/j['task']/'42/dynamic/metrics.json')['protocol']
                if j['variant']!='point':assert q['cohort_sha256']==baseline['cohort_sha256'],'Cohort changed'
                else:
                    canonical=lambda q:{k:sorted((r['sample'],r['cell'],r['target']) for r in seq) for k,seq in q['cohort'].items()}
                    assert canonical(q)==canonical(baseline),'Point-only cohort changed'
        assert read(out/'status.json')['state']=='complete' and (out/'final.pt').is_file()
    return sha(j['expected'])


def summarize(root,jobs):
    done={j['id'] for j in jobs if j['state']=='complete'};rows=[]
    for j in jobs:
        if not j['endpoint'] or j['id'] not in done:continue
        if j['task']=='tlp' and f"{j['stage']}/tlp/audit" not in done:continue
        d=read(j['expected']);task=j['task']
        metric,value=('macro_f1',d['validation']['macro_f1']) if task=='h01' else (
            ('mae',d['results']['val']['mae']) if task=='cell' else (
            ('f1',d['test']['f1']) if task=='ted' else ('missing_edge_macro_ap',d['test']['missing_edge_macro_ap'])))
        rows.append(dict(stage=j['stage'],task=task,variant=j['variant'],seed=42,metric=metric,value=value,
            path=j['expected'],sha256=sha(j['expected']),cohort_sha256=d['protocol'].get('cohort_sha256')))
    save(root/'summary.json',dict(expected_new_endpoints=14,completed_endpoints=len(rows),rows=rows,
        baseline_file=str(root/'v14_baselines.json'),interpretation='Single-seed development screen, not stability/significance evidence'))


def restore_snapshot(root,archive,sources,allow_active=False):
    """Explicit scheduling-only migration; optionally adopt live workers."""
    old=read(archive/'execution.json');status=read(archive/'status.json')
    assert status['state'] in ('waiting_resources','running'),'Unexpected queue state'
    assert allow_active or (not status['active'] and status['state']=='waiting_resources'),'Not an idle queue'
    assert not helper.process(old['pid']),'Old dispatcher is still alive'
    assert not (root/'failure.json').exists(),'Scheduler failure requires separate review'
    assert sha(root/'execution.json')==sha(archive/'execution.json'),'Ledger changed since snapshot'
    assert sha(root/'plan.json')==sha(archive/'plan.json'),'Plan changed since snapshot'
    allowed={str(Path(__file__).parent/name) for name in ('queue.py','test_queue.py')}
    allowed.add(str(HOME/'verification/queue_final.json'))
    assert sources.keys()==old['sources'].keys(),'Source set changed'
    changed={p for p,d in sources.items() if old['sources'][p]!=d}
    assert changed<=allowed,'Non-scheduling source changed: '+str(changed-allowed)
    jobs=make_jobs(root)
    assert len(jobs)==len(old['jobs'])
    for new,prev in zip(jobs,old['jobs']):
        assert all(new[k]==prev[k] for k in new if k!='state'),'Job specification changed'
        assert prev['state'] in ('complete','pending','running'),'Failed job requires separate review'
        if prev['state']=='running':
            assert allow_active and helper.same_process(prev['identity']),'Running worker is not adoptable'
        else: helper.assert_no_worker(Path(new['out']))
        if prev['state']=='complete':
            assert validate(prev)==prev['result_sha256'],'Completed result drift'
        elif prev['state']=='pending':
            assert not Path(new['out']).exists(),'Pending job output already exists'
            assert not (root/'logs'/(new['id'].replace('/','_')+'.log')).exists(),'Pending job log already exists'
    return old['jobs'],dict(archive=str(archive),previous_pid=old['pid'],previous_started_unix=old['started_unix'],
        changed_sources=sorted(changed),policy='resource_admission_without_prior_queue_barrier',adopted_active=allow_active)


restore_idle=lambda root,archive,sources: restore_snapshot(root,archive,sources,False)


class AdoptedProcess:
    def __init__(self,identity):self.identity=identity
    def poll(self):return None if helper.same_process(self.identity) else 0


def run(root,resume_idle_from=None,resume_active_from=None):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    archive=resume_active_from or resume_idle_from
    assert archive or not (root/'execution.json').exists(),'Refuse duplicate/automatic restart'
    sources=collect_sources();migration=None
    if resume_active_from:jobs,migration=restore_snapshot(root,resume_active_from,sources,True)
    elif resume_idle_from:jobs,migration=restore_idle(root,resume_idle_from,sources)
    else:
        jobs=make_jobs(root)
        assert all(not Path(j['out']).exists() for j in jobs),'Refuse existing output'
    snapshot_baselines(root)
    save(root/'plan.json',dict(jobs=jobs,sources=sources,seed=42,expected_new_endpoints=14,
        scientific_execution_stages=21,verification_stages=len(jobs)-21,migration=migration))
    children={j['id']:AdoptedProcess(j['identity']) for j in jobs if j['state']=='running'}
    started=time.time();last_launch=0
    def persist(state,blocked=None,resources=None):
        save(root/'execution.json',dict(pid=os.getpid(),started_unix=started,sources=sources,jobs=jobs,migration=migration))
        counts=dict(collections.Counter(j['state'] for j in jobs))
        save(root/'status.json',dict(state=state,counts=counts,pid=os.getpid(),updated_unix=time.time(),
            active=[dict(id=j['id'],pid=j['identity']['pid']) for j in jobs if j['state']=='running'],
            blocked=blocked or [],resources={k:v for k,v in (resources or {}).items() if k!='processes'}))
        summarize(root,jobs)
    try:
        persist('starting')
        while True:
            freeze(sources)
            for j in jobs:
                if j['state']!='running':continue
                rc=children[j['id']].poll()
                if rc is None:continue
                try:
                    assert rc==0,'Worker exit '+str(rc)
                    digest=validate(j);j.update(state='complete',result_sha256=digest,completed_unix=time.time())
                    print('COMPLETE',j['id'],flush=True)
                except Exception as exc:j.update(state='failed',error=repr(exc),exit_code=rc)
            if any(j['state']=='failed' for j in jobs):persist('failed_workers_preserved');return
            if all(j['state']=='complete' for j in jobs):persist('complete');return
            done={j['id'] for j in jobs if j['state']=='complete'}
            active=[j for j in jobs if j['state']=='running'];r=helper.resources();blocked=[]
            ready=[j for j in jobs if j['state']=='pending' and set(j['deps'])<=done]
            ready.sort(key=lambda j:({'audit':0,'fit':0,'smoke_fit':0,'extract':1,'smoke_extract':1}.get(j['kind'],2),j['variant']=='point'))
            if time.time()-last_launch>=60:
                for j in ready:
                    ok,reason=admission(j,r,active)
                    if not ok:blocked.append(dict(id=j['id'],reason=reason));continue
                    out=Path(j['out']);assert not out.exists();helper.assert_no_worker(out)
                    log=root/'logs'/(j['id'].replace('/','_')+'.log');log.parent.mkdir(exist_ok=True)
                    budget=profile(j)['gpu'];fraction=min(.32,budget/r['gpu_total_mib']) if budget else .08
                    env=dict(os.environ,PYTHONPATH=str(OLD),CUBLAS_WORKSPACE_CONFIG=':4096:8',CPSF_GPU_FRACTION=str(fraction),
                        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                    with log.open('x') as f:child=subprocess.Popen(j['command'],cwd=HOME,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    children[j['id']]=child;identity=helper.process(child.pid)
                    assert identity,'Missing process identity'
                    j.update(state='running',identity=identity,log=str(log),started_unix=time.time())
                    last_launch=time.time();print('START',j['id'],child.pid,flush=True);break
            persist('running' if active or any(j['state']=='running' for j in jobs) else 'waiting_resources',blocked[:20],r)
            time.sleep(60)
    except Exception as exc:
        persist('scheduler_failed_workers_preserved');save(root/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-idle-from',type=Path,help='Explicit archived idle ledger; scheduling-only migration')
    p.add_argument('--resume-active-from',type=Path,help='Explicit archived active ledger; adopt live workers')
    a=p.parse_args();run(a.out.resolve(),a.resume_idle_from.resolve() if a.resume_idle_from else None,
        a.resume_active_from.resolve() if a.resume_active_from else None)

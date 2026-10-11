"""Queue planning and admission tests; launches no workers."""
import argparse,tempfile,copy
from types import SimpleNamespace
from unittest.mock import patch
from . import queue as q


def main(out):
    root=q.HOME/'experiments';jobs=q.make_jobs(root);by={j['id']:j for j in jobs}
    assert sum(j['endpoint'] for j in jobs)==14
    science=[j for j in jobs if j['kind'] in ('train','extract','fit','audit')]
    assert len(science)==21
    for j in science:
        if '--seed' in j['command']:assert j['command'][j['command'].index('--seed')+1]=='42'
    for task in ('h01','ted','tlp'):
        assert {'residual/h01/train','residual/ted/train','residual/tlp/train','residual/tlp/audit','residual/cell/fit'}<=set(by[f'routing/{task}/train']['deps'])
        assert {'routing/h01/train','routing/ted/train','routing/tlp/train','routing/tlp/audit','routing/cell/fit'}<=set(by[f'role/{task}/train']['deps'])
    # Check DAG reaches every job without skipping a dependency.
    done=set()
    while len(done)<len(jobs):
        ready={j['id'] for j in jobs if j['id'] not in done and set(j['deps'])<=done}
        assert ready,'Dependency cycle';done|=ready
    r=dict(gpu_total_mib=24564,gpu_free_mib=22000,ram_available_mib=28000,gpu_by_pid={},processes={})
    with patch.object(q,'read',side_effect=AssertionError('Admission must not inspect prior completion')), \
         patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=20*2**30)):
        assert q.admission(by['residual/h01/train'],r,[])[0]
        assert q.admission(by['residual/h01/train'],dict(r,gpu_free_mib=1000),[])[1]=='gpu_budget'
        assert q.admission(by['stress/cell/role'],dict(r,ram_available_mib=12000),[])[1]=='ram_budget'
        active=[dict(by['stress/cell/role'],identity={'pid':1234})]
        assert q.admission(by['stress/cell/point'],r,active)[1]=='heavy_extractor_limit'
        assert q.admission(by['residual/h01/train'],r,active*3)[1]=='managed_slots'
        assert q.admission(by['residual/h01/train'],dict(r,gpu_by_pid={i:100 for i in range(4)}),[])[1]=='global_gpu_slots'
        # Two old TLP workers: reserve growth to 8 GiB each, not just measured use.
        shared=dict(r,gpu_free_mib=15691,gpu_by_pid={11:4524,12:3852},processes={
            i:dict(cmd='python -m tools.cpsf_v13.train --task tlp --variant dynamic ',rss_mib=7168) for i in (11,12)})
        assert q.admission(by['smoke/dynamic/cell_extract'],shared,[])[0]
        assert q.admission(by['stress/local/residual'],shared,[])[1]=='gpu_budget'
        external=dict(r,processes={11:dict(cmd='python -m tools.cpsf_v14.cell --stage extract ',rss_mib=7168)})
        assert q.admission(by['stress/cell/role'],external,[])[1]=='heavy_extractor_limit'
        with patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=7*2**30)):
            assert q.admission(by['residual/h01/train'],r,[])[1]=='disk_reserve'
    # Completed work can be reused only under an explicit, idle, source-locked migration.
    with tempfile.TemporaryDirectory() as tmp, patch.object(q,'HOME',q.Path(tmp)/'home'):
        root=q.Path(tmp)/'run';archive=q.Path(tmp)/'archive'
        oldjobs=q.make_jobs(root);oldjobs[0].update(state='complete',result_sha256='result')
        sources={str(q.Path(q.__file__)):'new',str(q.HOME/'encoder.py'):'unchanged'}
        old=dict(pid=123,started_unix=1,sources=dict(sources),jobs=oldjobs)
        old['sources'][str(q.Path(q.__file__))]='old'
        for directory in (root,archive):
            q.save(directory/'execution.json',old);q.save(directory/'plan.json',{'jobs':oldjobs})
        q.save(archive/'status.json',dict(state='waiting_resources',active=[]))
        with patch.object(q.helper,'process',return_value=None), \
             patch.object(q.helper,'assert_no_worker'),patch.object(q,'validate',return_value='result'):
            restored,meta=q.restore_idle(root,archive,sources)
            assert restored==oldjobs and meta['previous_pid']==123
            def rejected():
                try:q.restore_idle(root,archive,sources)
                except AssertionError:return
                raise AssertionError('Unsafe resume accepted')
            with patch.object(q.helper,'process',return_value={'pid':123}):rejected()
            with patch.object(q,'validate',return_value='wrong'):rejected()
            altered=dict(sources);altered[str(q.HOME/'encoder.py')]='drift'
            try:q.restore_idle(root,archive,altered)
            except AssertionError:pass
            else:raise AssertionError('Core drift accepted')
            for state in ('running','failed'):
                unsafe=copy.deepcopy(old);unsafe['jobs'][1]['state']=state
                for directory in (root,archive):q.save(directory/'execution.json',unsafe)
                rejected()
            for directory in (root,archive):q.save(directory/'execution.json',old)
            unsafe=copy.deepcopy(old);unsafe['jobs'][1]['deps']=[];unsafe['jobs'][1]['command']=['changed']
            for directory in (root,archive):q.save(directory/'execution.json',unsafe)
            rejected()
    q.save(out,dict(passed=True,new_endpoints=14,scientific_execution_stages=21,verification_stages=len(jobs)-21,
        planned_jobs=len(jobs),gates=['seed42_only','four_task_stage_barriers','DAG','prior_jobs_preserved',
            'gpu_ram_guards','one_heavy_extractor','disk_guard','global_and_managed_slots',
            'external_worker_growth_reservation','no_prior_queue_barrier','explicit_idle_resume_source_and_result_validation'],
        source_sha256={'queue.py':q.sha(q.Path(q.__file__)),'test_queue.py':q.sha(q.Path(__file__))}))
    print('QUEUE_TESTS_PASSED',len(jobs),'jobs / 14 scientific endpoints',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=q.Path,required=True);main(p.parse_args().out)

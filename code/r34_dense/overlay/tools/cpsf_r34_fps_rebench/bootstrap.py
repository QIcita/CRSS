"""Freeze a six-job extension. Parent protocols and datasets are read-only."""
import shutil
from .common import ROOT,PARENT,FINE,OLD_R34,PARENT_SHA,FINE_SHA,OLD_SHA,TASKS,read,sha,save

def main():
    assert not (ROOT/'protocol.json').exists(),'Do not overwrite frozen experiments'
    assert sha(PARENT/'protocol.json')==PARENT_SHA and sha(FINE/'protocol.json')==FINE_SHA
    assert sha(OLD_R34/'protocol.json')==OLD_SHA
    parent=read(PARENT/'protocol.json');fine=read(FINE/'protocol.json')
    old_source=OLD_R34/'tools/cpsf_path_message/model.py'
    old=read(OLD_R34/'protocol.json')
    assert sha(old_source)==old['source_sha256']['tools/cpsf_path_message/model.py']
    shutil.copyfile(old_source,ROOT/'tools/cpsf_r34_fps_rebench/frozen_r34.py')
    # Only exact source and metadata copies. Geometric files are never copied.
    shutil.copyfile(FINE/'labels.json',ROOT/'labels.json')
    tasks=dict(parent['tasks']);tasks.update(fine['tasks'])
    for task in TASKS:
        source=FINE if task=='flywire_fine' else PARENT
        dest=ROOT/'data'/task;dest.parent.mkdir(exist_ok=True)
        dest.symlink_to(source/'data'/task,target_is_directory=True)
        (ROOT/'orders').mkdir(exist_ok=True)
        shutil.copyfile(source/'orders'/(task+'.json'),ROOT/'orders'/(task+'.json'))
    files=dict(fine['files']);files[str(old_source)]=sha(old_source)
    for f in [PARENT/'protocol.json',FINE/'protocol.json',OLD_R34/'protocol.json',ROOT/'labels.json']:
        files[str(f)]=sha(f)
    for f in list((ROOT/'tools/cpsf_r34_fps_rebench').glob('*.py'))+list((ROOT/'orders').glob('*.json')):
        files[str(f)]=sha(f)
    for f,h in files.items():assert sha(f)==h,f
    save(ROOT/'protocol.json',dict(version=ROOT.name,files=files,seed=42,pretrained=False,test_used=False,
        tasks=tasks,jobs=[dict(id='r34_'+t,model='r34',task=t) for t in TASKS],
        parents={str(PARENT):PARENT_SHA,str(FINE):FINE_SHA,str(OLD_R34):OLD_SHA},
        architecture='R34 A2 stage1 path_message exactly, 16D/6608 parameters; A3 R28 shared backbone and existing task heads',
        stages_2_and_3_enabled=False,geometry='Exact parent manifest/cache/sidecar reuse; no resampling or new features',
        training='Parent optimizer/loss/budget/orders/final epoch; all tasks fresh seed42 initialization',
        selection='fixed final epoch; development only',
        promotion=dict(current_reference='R28',candidate='R34',rule='All six matched primary metrics non-worse, at least one improves beyond 1e-12; tradeoffs keep R28 pending review',
            primary=dict(h01_local='macro_f1',h01_global='mae',flywire_coarse='macro_f1',flywire_fine='macro_f1',ted='f1',tlp='full_adjacency_macro_ap'),
            secondary_tlp='missing_edge_macro_ap',test_feedback=False),
        concurrency='one additional R34 worker; account for A3 and fine-task reservations, RAM/disk guards',
        worker_source=str(PARENT/'tools/cpsf_geometry_rebench/worker.py'),
        worker_changes=['R34 identity and message gradient/compatibility gates','fine-classification metrics from frozen fine task','input readiness audit'],
        source_of_message=str(old_source)))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)

if __name__=='__main__':main()

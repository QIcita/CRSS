import shutil
from .common import ROOT,REFERENCE,INPUT,TASKS,REFERENCE_SHA,INPUT_SHA,read,save,sha,AP,AP_SHA,variant

def main():
    assert not (ROOT/'protocol.json').exists(),'Immutable version already frozen'
    assert sha(REFERENCE/'protocol.json')==REFERENCE_SHA and sha(INPUT/'protocol.json')==INPUT_SHA
    parent=read(REFERENCE/'protocol.json');files=dict(read(AP/'protocol.json')['files']);manifests={}
    for task in TASKS:
        path=INPUT/'data'/task/'manifest.json';rr=read(path)['records'];cfg=parent['tasks'][task]
        tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
        assert (len(tr),len(va))==(cfg['train'],cfg['val'])
        assert len(tr)+len(va)==len(rr) and len({r['key'] for r in rr})==len(rr)
        assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
        for r in rr:
            assert r['task']==task and r['protocol_sha256']==INPUT_SHA
            for field in ['path','feature_path','graph_path']:assert __import__('pathlib').Path(r[field]).is_file()
        manifests[task]=sha(path)
        dest=ROOT/'data'/task;dest.parent.mkdir(exist_ok=True);dest.symlink_to(INPUT/'data'/task,target_is_directory=True)
        (ROOT/'orders').mkdir(exist_ok=True)
        shutil.copyfile(REFERENCE/'orders'/(task+'.json'),ROOT/'orders'/(task+'.json'))
        save(ROOT/'input_audits'/(task+'.json'),dict(passed=True,manifest_sha256=manifests[task],train=len(tr),val=len(va),
             cell_disjoint=True,canonical_fps_unchanged=True,test_used=False,per_cache_bytes_verified_on_first_use=True,
             full_byte_reaudit_before_training=False,parent_identity_reused=True))
    for f in list((ROOT/'tools/cpsf_r34_patch_repair').glob('*.py'))+list((ROOT/'orders').glob('*.json')):
        files[str(f)]=sha(f)
    files[str(REFERENCE/'protocol.json')]=REFERENCE_SHA
    files[str(AP/'protocol.json')]=AP_SHA
    files[str(ROOT/'variant.json')]=sha(ROOT/'variant.json')
    save(ROOT/'protocol.json',dict(version=ROOT.name,seed=42,files=files,input_manifests=manifests,
        tasks={t:parent['tasks'][t] for t in TASKS},jobs=[dict(model='r34',task=t) for t in TASKS],
        input_root=str(INPUT),reference_root=str(REFERENCE),reference_protocol_sha256=REFERENCE_SHA,
        repair_variant=variant(),policy=dict(min_patch_size=.5,scale_multiplier=2,target_mean_points=10,max_levels=25,
                    no_task_condition=True,source='same canonical FPS points and source-derived origin',
                    source_mass_preserved=True,all_input_points_used=True,context_voxel_size=.5,
                    skeleton_arclength_and_relation_radius_unchanged=True,rebuild_patch_region_associations=variant()=='context_control',
                    actual_patch_size_in_scale_encoding=variant()=='context_control',
                    original_utonia_input=True,original_fine_support=variant()=='support_preserved',
                    dense_neighborhood_geometry_unit=.5),
        model='R34 unchanged parameters: isolate Utonia input restoration or dense SPoTr neighborhoods with original fine statistics and RCM support',
        training='Exact parent optimizer, schedule, losses, seed42, epoch orders, update budgets and fixed final epoch',
        test_used=False,pretrained=False,validation_selection=False,baseline_inputs_untouched=True,
        acceptance='Fixed0.5 two-update exact parent replay; extreme-shape backward; actual-scale hook; finite gradients; A-B-A; two optimizer updates',
        cache='Read-only canonical and AP10 caches, bounded RAM only; hash verified on every uncached read; no labels in geometry',
        comparison='New version separate from R34 A2, no mixing old scores into new version'))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)

if __name__=='__main__':main()

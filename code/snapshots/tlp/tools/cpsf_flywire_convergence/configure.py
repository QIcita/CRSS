import json
from .common import ROOT,COHORT,ARMS,save,sha,digest_object,verify_sources

def main():
    assert not (ROOT/'protocol.json').exists()
    parent=json.loads((COHORT/'protocol.json').read_text());verify_sources(COHORT,parent)
    # No historical source is overwritten to construct this experiment.
    verify_sources(ROOT,parent)
    ready=json.loads((COHORT/'cache/READY.json').read_text())
    assert ready['ready'] and ready['old_split_preserved']
    manifest=json.loads((COHORT/'cache/manifest.json').read_text())
    assert sha(COHORT/'cache/manifest.json')==parent['cache_manifest_sha256']
    refs={}
    for arm in ARMS:
        p=COHORT/'results'/('coarse_'+arm);m=json.loads((p/'metrics.json').read_text())
        assert m['protocol_sha256']==sha(COHORT/'protocol.json') and m['final_sha256']==sha(p/'final.pt')
        assert m['seed']==42 and not m['test_used'] and not m['smoke'] and m['total_optimizer_steps']==62200
        refs[arm]=dict(path=str(p),sha256={k:sha(p/k) for k in ('initial.pt','final.pt','metrics.json')},
            val=m['val'],val_legacy=m['val_legacy'],val_added=m['val_added'])
    architecture_files={k:v for k,v in parent['source_sha256'].items()
        if k.startswith(('tools/cpsf_v21_segment_units/','tools/cpsf_v22_shared_region/',
            'tools/cpsf_v23_residual_region/','tools/cpsf_v23_scale_ablation/',
            'tools/cpsf_flywire_v23/','tools/cpsf_h01_topology_repair/')) and k.endswith(('model.py','data.py','adapter.py'))}
    assert architecture_files
    p=dict(parent)
    p.update(version='FlyWire V23 expanded cohort unweighted CE control; not V24/V25/V26',
        experiment_id=ROOT.name,arms=list(ARMS),reference_runs=refs,
        reference_policy='same expanded cohort weighted-CE final checkpoints; exact legacy and expanded reproduction',
        inherited_protocol_sha256=sha(COHORT/'protocol.json'),
        architecture_files=architecture_files,architecture_sha256=digest_object(architecture_files),
        split_sha256=digest_object(sorted((r['root_id'],r['split']) for r in manifest['records'])),
        label_sha256=digest_object(manifest['labels']),data_root=str(COHORT),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FLYWIRE_CONVERGENCE_PROTOCOL_20261003.md'),
        loss=dict(parent['loss'],name='unweighted smoothed cross entropy',
            formula='mean_i CE_smooth(logits_i,y_i)',class_weights=[1.]*6),
        test_policy='validation only; never open test predictions for convergence decisions')
    save(ROOT/'smoke_records.json',json.loads((COHORT/'smoke_records.json').read_text()))
    p['smoke_records_sha256']=sha(ROOT/'smoke_records.json')
    save(ROOT/'protocol.json',p)

if __name__=='__main__':main()

"""Read direct run outputs only. Nested baseline/reference metrics are never current results."""
import json,datetime
from .common import ROOT,BASE,save,sha,verify_sources

CATALOG={
 'CPSF_V23_RESIDUAL_REGION_20261001':('H01','V23 residual region','original coarse hierarchy'),
 'CPSF_V23_SCALE_ABLATION_20261001':('H01','V23 encoding/resolution ablations','original coarse hierarchy'),
 'CPSF_V24_PHYSICAL_CONDITIONING_20261001':('H01','V24; hierarchy_only is V23-code control','minimum-four hierarchy runtime adapter'),
 'CPSF_V25_POSITION_ROUTING_20261001':('H01','V25; baseline is repaired V23-code control','minimum-four hierarchy runtime adapter'),
 'CPSF_FLYWIRE_V23_20261001':('FlyWire8121','V23 FlyWire','original coarse hierarchy'),
 'CPSF_FLYWIRE_V23_BALANCED_20261002':('FlyWire8121','V23 FlyWire; balanced loss','original coarse hierarchy'),
 'CPSF_FLYWIRE_COARSE_COHORT_20261002':('FlyWire9883','V23 FlyWire; balanced loss','original coarse hierarchy'),
 'CPSF_FLYWIRE_V23_CE_EXPANDED_20261003':('FlyWire9883','V23 FlyWire; ordinary CE','original coarse hierarchy'),
}

def main():
    experiments=[]
    for name,(dataset,model,hierarchy) in CATALOG.items():
        root=BASE/name
        if not (root/'protocol.json').exists():continue
        p=json.loads((root/'protocol.json').read_text());verify_sources(root,p)
        record=dict(experiment_id=name,root=str(root),dataset=dataset,model=model,hierarchy=hierarchy,
            protocol_sha256=sha(root/'protocol.json'),source_hashes_verified=True,
            seed=p['seed'],loss=p.get('loss',p.get('local_objective','see protocol')),
            counts=p.get('counts'),runs=[])
        # Exactly one directory below results: do not ingest frozen_reference or protocol.baselines.
        for path in sorted((root/'results').glob('*/metrics.json')):
            m=json.loads(path.read_text());out=path.parent
            if m.get('smoke'):continue
            entry=dict(run_id=name+'/'+out.name,result_origin=str(path),metrics_sha256=sha(path),
                task=m.get('task'),arm=m.get('arm',out.name),val=m.get('val'),
                val_legacy=m.get('val_legacy'),val_added=m.get('val_added'),
                checkpoint_sha256=sha(out/'final.pt') if (out/'final.pt').exists() else None,
                initial_sha256=sha(out/'initial.pt') if (out/'initial.pt').exists() else None,
                metric_protocol_link_present='protocol_sha256' in m,
                metric_checkpoint_link_present='final_sha256' in m)
            if 'protocol_sha256' in m:assert m['protocol_sha256']==record['protocol_sha256'],str(path)
            if 'final_sha256' in m:assert m['final_sha256']==entry['checkpoint_sha256'],str(path)
            if 'seed' in m:assert m['seed']==42
            if entry['checkpoint_sha256'] is None:raise RuntimeError('Completed metrics without final checkpoint: '+str(path))
            record['runs'].append(entry)
        experiments.append(record)
    save(ROOT/'version_registry.json',dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        test_predictions_read=False,policy='Direct originating runs only; no relabeling historical controls as candidate results',
        experiments=experiments))

if __name__=='__main__':main()

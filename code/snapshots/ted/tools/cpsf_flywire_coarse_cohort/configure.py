import json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
from tools.cpsf_flywire_balanced.loss import weights_from_counts
from .plan import ROOT,OLD,BAL,ARMS


def main():
    assert not (ROOT/'protocol.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert json.loads((BAL/'pipeline_status.json').read_text())['state']=='complete'
    base=json.loads((BAL/'protocol.json').read_text());ready=json.loads((ROOT/'cache/READY.json').read_text())
    assert ready['ready'] and ready['old_split_preserved']
    assert sha(ROOT/'cache/manifest.json')==ready['manifest_sha256']
    manifest=json.loads((ROOT/'cache/manifest.json').read_text());train=[r for r in manifest['records'] if r['split']=='train']
    counts=[sum(r['coarse_target']==i for r in train) for i in range(6)]
    reference={}
    for arm in ARMS:
        p=BAL/'results'/('coarse_'+arm);m=json.loads((p/'metrics.json').read_text())
        assert not m['smoke'] and not m['test_used'] and m['epochs']==50
        reference[arm]=dict(path=str(p),sha256={k:sha(p/k) for k in ('initial.pt','final.pt','metrics.json')},val=m['val'])
    smoke=[min((r for r in train if not r['legacy'] and r['coarse_target']==i),key=lambda r:r['root_id']) for i in range(6)]
    largest=json.loads((OLD/'contracts.json').read_text())['largest_training_root']
    smoke.append(next(r for r in train if r['root_id']==largest))
    smoke.append(max((r for r in train if not r['legacy']),key=lambda r:r['relation_pairs']))
    smoke=list({r['root_id']:r for r in smoke}.values());save(ROOT/'smoke_records.json',smoke)
    base.update(version='FlyWire coarse cohort expansion; matched sample/update budget',
        counts=ready['counts'],epoch_cells=4975,optimizer_steps=62200,train_pool=len(train),
        reference_runs=reference,reference_policy='frozen balanced models; same legacy and expanded validation',
        loss=dict(base['loss'],train_counts=counts,class_weights=weights_from_counts(counts).tolist()),
        cache_manifest_sha256=ready['manifest_sha256'],smoke_records_sha256=sha(ROOT/'smoke_records.json'),
        source_sha256=plan['source_sha256'],plan_sha256=sha(ROOT/'plan.json'),
        inherited_protocol_sha256=sha(BAL/'protocol.json'),largest_training_root=largest,
        document_sha256=sha(ROOT/'CPSF_FLYWIRE_COARSE_COHORT_PROTOCOL_20261002.md'),
        test_policy='validation only; old test split preserved; no expanded test evaluation',
        sampling='each budget round: seed42 permutation of full train pool, take first4975; same sequence for all arms')
    save(ROOT/'protocol.json',base)


if __name__=='__main__':main()

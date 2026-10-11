import json
from collections import Counter
from pathlib import Path
from tools.cpsf_v21_segment_units.io import sha, save
from .loss import weights_from_counts

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/root/megrez-tmp/CPSF_FLYWIRE_V23_20261001')
ARMS = ('full', 'mean_only', 'no_fusion')


def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((BASE/'protocol.json').read_text())
    for name,digest in old['source_sha256'].items():
        assert sha(BASE/name)==digest and sha(ROOT/name)==digest,name
    ready=json.loads((ROOT/'cache/READY.json').read_text())
    assert ready['ready'] and sha(ROOT/'cache/manifest.json')==ready['manifest_sha256']
    manifest=json.loads((ROOT/'cache/manifest.json').read_text())
    rows=[r for r in manifest['records'] if r['variant']=='joint']
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    assert (len(train),len(val))==(4975,1573)
    assert not ({r['root_id'] for r in train}&{r['root_id'] for r in val})
    labels=manifest['labels']['coarse'];freq=Counter(r['coarse_target'] for r in train)
    counts=[freq[i] for i in range(len(labels))];weights=weights_from_counts(counts).tolist()
    references={}
    for arm in ARMS:
        p=BASE/'results'/('coarse_'+arm);m=json.loads((p/'metrics.json').read_text())
        assert m['seed']==42 and m['epochs']==50 and not m['test_used']
        assert m['protocol_sha256']==sha(BASE/'protocol.json')
        assert m['cache_manifest_sha256']==ready['manifest_sha256']
        references[arm]=dict(path=str(p),sha256={f:sha(p/f) for f in ('initial.pt','final.pt','metrics.json')},val=m['val'])
    largest=json.loads((BASE/'contracts.json').read_text())['largest_training_root']
    smoke=[min((r for r in train if r['coarse_target']==i),key=lambda r:r['root_id']) for i in range(len(labels))]
    smoke.append(next(r for r in train if r['root_id']==largest))
    smoke=list({r['root_id']:r for r in smoke}.values())
    save(ROOT/'smoke_records.json',smoke)
    save(ROOT/'protocol.json',dict(version='FlyWire V23 coarse class-balanced objective validation',
        tasks=['coarse'],arms=ARMS,seed=42,epochs=50,batch=4,lr=.0003,weight_decay=.01,
        label_smoothing=.05,lr_min=.00003,clip=5,selection='fixed final epoch',
        initialization=old['initialization'],counts=old['counts'],
        loss=dict(name='target-frequency sample-weighted smoothed cross entropy',
            formula='mean_i [N/(K*n_train[y_i]) * CE_smooth(logits_i,y_i)]',
            normalization='fixed training expectation of weight = 1; batch mean by cell count, never by sum of weights',
            train_counts=counts,class_weights=weights,class_names=labels,weight_source_split='train'),
        balanced_sampling=False,reference_runs=references,
        inherited_protocol_sha256=sha(BASE/'protocol.json'),cache_manifest_sha256=ready['manifest_sha256'],
        smoke_records_sha256=sha(ROOT/'smoke_records.json'),largest_training_root=largest,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FLYWIRE_BALANCED_PROTOCOL_20261002.md'),
        test_policy='validation-only supplementary experiment; do not run test or select checkpoints on test',
        decision='compare all three arms to same-arm unweighted baselines and to each other; no claim full must win'))


if __name__=='__main__':main()

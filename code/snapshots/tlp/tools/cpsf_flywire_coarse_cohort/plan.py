"""Immutable coarse-only expansion of the original 10K selection."""
import csv,json,hashlib,urllib.request
from collections import Counter
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha

ROOT=Path(__file__).resolve().parents[2]
OLD=Path('/root/megrez-tmp/CPSF_FLYWIRE_V23_20261001')
BAL=Path('/root/megrez-tmp/CPSF_FLYWIRE_V23_BALANCED_20261002')
DATA=Path('/root/megrez-tmp/CPSF_FLYWIRE_TASKS_20260930')
ARMS=('full','mean_only','no_fusion')


def main():
    assert not (ROOT/'plan.json').exists()
    original=json.loads((OLD/'protocol.json').read_text())
    for name,digest in original['source_sha256'].items():assert sha(ROOT/name)==digest and sha(OLD/name)==digest,name
    ready=json.loads((DATA/'dataset/READY.json').read_text())
    assert sha(DATA/'dataset/frozen_manifest.jsonl')==ready['manifest_sha256']
    old={r['root_id']:r for r in (json.loads(l) for l in (DATA/'dataset/frozen_manifest.jsonl').read_text().splitlines())}
    classes=json.loads((DATA/'dataset/labels.json').read_text())['coarse']
    source=ROOT/'original_10k.csv';rows=list(csv.DictReader(source.open()));assert len(rows)==len({r['root_id'] for r in rows})==10000
    audit=json.loads((DATA/'metadata/annotation_source_audit.json').read_text());annpath=ROOT/'source_annotations.tsv'
    if not annpath.exists():
        with urllib.request.urlopen(audit['url'],timeout=120) as response:annpath.write_bytes(response.read())
    assert sha(annpath)==audit['sha256']
    ann={r['root_id']:r for r in csv.DictReader(annpath.open(),delimiter='\t')}
    excluded_ids={r['root_id'] for r in json.loads((DATA/'dataset/excluded.json').read_text())}
    kept=[]
    for r in rows:
        a=ann[r['root_id']];assert not a['status']
        assert all(a[k]==r[k] for k in ('cell_type','super_class','cell_class'))
        if r['super_class'] not in classes or r['root_id'] in excluded_ids:continue
        if r['root_id'] in old:
            o=old[r['root_id']];assert all(r[k]==o[k] for k in ('cell_type','super_class','vfb_id'))
            kept.append(dict(o,legacy=True,fine_available=True))
        else:kept.append(dict(r,legacy=False,fine_available=False,fine_target=-1,coarse_target=classes.index(r['super_class'])))
    for c in classes:
        group=sorted([r for r in kept if not r['legacy'] and r['super_class']==c],key=lambda r:hashlib.sha256(('coarse-expansion42|'+r['root_id']).encode()).hexdigest())
        nv=len(group)//5;nt=len(group)//5;ntr=len(group)-nv-nt
        for i,r in enumerate(group):r['split']='train' if i<ntr else 'val' if i<ntr+nv else 'test'
    kept.sort(key=lambda r:r['root_id']);assert sum(r['legacy'] for r in kept)==8121
    assert len(kept)==9883 and sum(not r['legacy'] for r in kept)==1762
    assert all(r['split']==old[r['root_id']]['split'] for r in kept if r['legacy'])
    groups={s:{r['root_id'] for r in kept if r['split']==s} for s in ('train','val','test')}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    save(ROOT/'candidates.json',kept)
    pilot=[min((r for r in kept if not r['legacy'] and r['split']=='train' and r['super_class']==c),key=lambda r:r['root_id'])['root_id'] for c in classes]
    save(ROOT/'plan.json',dict(seed=42,coarse_labels=classes,candidates=9883,legacy=8121,new=1762,
        counts={s:len(groups[s]) for s in groups},per_class={c:dict(Counter(r['split'] for r in kept if r['super_class']==c)) for c in classes},
        old_split_preserved=True,fine_task_changed=False,new_fine_target=-1,pilot_new_training_roots=pilot,
        original_10k_sha256=sha(source),annotation_sha256=sha(annpath),candidate_sha256=sha(ROOT/'candidates.json'),
        old_manifest_sha256=ready['manifest_sha256'],old_protocol_sha256=sha(OLD/'protocol.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FLYWIRE_COARSE_COHORT_PROTOCOL_20261002.md'),
        freeze_gate='all attempted; >=95% new retained in each coarse class; no silently moving splits after QA',
        evaluation='fixed six classes; legacy validation anchor plus all expanded validation; no test evaluation'))


if __name__=='__main__':main()

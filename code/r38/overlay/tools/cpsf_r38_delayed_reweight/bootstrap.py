import shutil
from .common import ROOT, DENSE, DENSE_SHA, INPUT, TASK, LABELS, read, save, sha

def main():
    assert not (ROOT/'protocol.json').exists()
    assert sha(DENSE/'protocol.json') == DENSE_SHA
    parent = read(DENSE/'protocol.json')
    rr = read(INPUT/'data'/TASK/'manifest.json')['records']
    tr = [r for r in rr if r['split'] == 'train']; va = [r for r in rr if r['split'] == 'val']
    cfg = parent['tasks'][TASK]
    assert (len(tr), len(va)) == (cfg['train'], cfg['val']) == (6039, 1922)
    assert len({r['key'] for r in rr}) == len(rr)
    assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
    label_path = INPUT.parent/'CPSF_FLYWIRE_TASKS_20260930/dataset/labels.json'
    assert read(label_path)['coarse'] == LABELS
    counts = [sum(r['target'] == c for r in tr) for c in range(6)]
    assert counts == [102, 962, 3950, 766, 28, 231]
    u = [(len(tr)/(6*n))**.25 for n in counts]
    mean = sum(n*x for n, x in zip(counts, u))/len(tr)
    weights = [x/mean for x in u]
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'data'/TASK).symlink_to(INPUT/'data'/TASK, target_is_directory=True)
    (ROOT/'orders').mkdir(exist_ok=True)
    shutil.copyfile(DENSE/'orders'/f'{TASK}.json', ROOT/'orders'/f'{TASK}.json')
    files = dict(parent['files'])
    files[str(DENSE/'protocol.json')] = DENSE_SHA
    files[str(label_path)] = sha(label_path)
    for f in list((ROOT/'tools/cpsf_r38_delayed_reweight').glob('*.py'))+list((ROOT/'orders').glob('*.json')):
        files[str(f)] = sha(f)
    for f, h in files.items(): assert sha(f) == h, ('source drift', f)
    manifest_sha = sha(INPUT/'data'/TASK/'manifest.json')
    p = dict(version=ROOT.name, seed=42, pretrained=False, test_used=False,
             files=files, tasks={TASK:cfg}, jobs=[dict(id='r38_'+TASK, model='r38', task=TASK)],
             input_manifests={TASK:manifest_sha}, parent=str(DENSE), parent_protocol_sha256=DENSE_SHA,
             architecture=parent['architecture'], neighborhood=parent['neighborhood'], fusion=parent['fusion'],
             training='Inherited frozen R38 loop; only target-class loss multiplier changes after epoch30',
             primary_metric='macro_f1', secondary_metrics=['accuracy','weighted_f1','per_class_precision_recall_f1'],
             selection='fixed final epoch50; no development checkpoint selection', prediction='argmax raw logits',
             loss=dict(kind='delayed mild target-class weighted smoothed CE', alpha=.25,
                       labels=LABELS, train_counts=counts, class_weights=weights,
                       normalization='train-frequency-weighted mean one', label_smoothing=.05,
                       warm_epochs=30, ramp_epochs=5, full_weight_epochs=[36,50],
                       ramp='0 if epoch<=30 else min((epoch-30)/5,1)',
                       reduction='Unweighted per-cell smoothed CE times effective target-class weight, divide by group cell count',
                       oversampling=False),
             unchanged=['FPS points','architecture','split','seed','initialization','sample orders','50 epochs','62200 updates',
                        'batch4','AdamW','learning rates','cosine schedule','weight decay','gradient clipping','label smoothing','final-epoch selection'],
             acceptance='CPU loss/gradient contracts; exact parent initialization and six real updates; active weighted-gradient and extreme-shape checks; discard all acceptance weights')
    save(ROOT/'protocol.json', p)
    save(ROOT/'input_audits'/f'{TASK}.json', dict(passed=True, manifest_sha256=manifest_sha,
         train=len(tr), val=len(va), cell_disjoint=True, canonical_input_unchanged=True,
         count_source='training records only', test_used=False))
    print('FROZEN', sha(ROOT/'protocol.json'), flush=True)

if __name__ == '__main__': main()

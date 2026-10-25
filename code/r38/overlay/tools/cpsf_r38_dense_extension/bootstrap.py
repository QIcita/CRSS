"""Freeze one new task, leaving the completed and live dense tasks intact."""
import hashlib
import shutil
import numpy as np
from pathlib import Path
from .common import ROOT, DENSE, R38, R34, INPUT, FINE, TASKS, PARENTS, manifest, rows, read, save, sha


def main():
    assert not (ROOT / 'protocol.json').exists(), 'Existing protocols are immutable'
    for root, expected in PARENTS.items():
        assert sha(root / 'protocol.json') == expected
    parent = read(R34 / 'protocol.json')
    files = dict(read(DENSE / 'protocol.json')['files'])
    files.update(parent['files'])
    manifests = {}
    for task in TASKS:
        records = rows(task)
        cfg = parent['tasks'][task]
        train = [r for r in records if r['split'] == 'train']
        val = [r for r in records if r['split'] == 'val']
        assert (len(train), len(val), cfg['classes']) == (4975, 1573, 148)
        assert len(records) == len(train) + len(val) == len({r['key'] for r in records})
        assert {r['cell'] for r in train}.isdisjoint(r['cell'] for r in val)
        assert set(r['target'] for r in records) == set(range(148))
        for row in records:
            assert row['task'] == task and row['protocol_sha256'] == PARENTS[FINE]
            assert row['target'] == row['fine_target']
            assert row['parent_protocol_sha256'] == PARENTS[INPUT]
            for key in ('path', 'feature_path', 'graph_path'):
                assert Path(row[key]).is_file()
        manifests[task] = sha(manifest(task))
        destination = ROOT / 'data' / task
        destination.parent.mkdir(exist_ok=True)
        destination.symlink_to(manifest(task).parent, target_is_directory=True)
        (ROOT / 'orders').mkdir(exist_ok=True)
        shutil.copyfile(R34 / 'orders' / (task + '.json'), ROOT / 'orders' / (task + '.json'))
        orders = read(ROOT / 'orders' / (task + '.json'))
        rng = np.random.RandomState(42)
        assert len(orders) == cfg['epochs'] == 50
        for epoch, order in enumerate(orders):
            keys = [train[i]['key'] for i in rng.permutation(len(train))[:cfg['round_cells']]]
            assert hashlib.sha256('\n'.join(keys).encode()).hexdigest() == order['sha256']
        save(ROOT / 'input_audits' / (task + '.json'), dict(
            passed=True, manifest_sha256=manifests[task], train=len(train), val=len(val),
            cell_disjoint=True, test_used=False, canonical_input_unchanged=True,
            cache_bytes_hash_verified_on_load=True, all_epoch_orders_exact=True))
    shutil.copyfile(R34 / 'labels.json', ROOT / 'labels.json')
    manifests['h01_local'] = sha(manifest('h01_local'))
    for root, expected in PARENTS.items():
        files[str(root / 'protocol.json')] = expected
    for path in list((ROOT / 'tools/cpsf_r38_dense_extension').glob('*.py')) + list((ROOT / 'orders').glob('*.json')) + [ROOT / 'labels.json']:
        files[str(path)] = sha(path)
    for path, expected in files.items():
        assert sha(path) == expected, path
    save(ROOT / 'protocol.json', dict(
        version=ROOT.name, files=files, seed=42, pretrained=False, test_used=False,
        tasks={task: parent['tasks'][task] for task in TASKS},
        jobs=[dict(id='r38_' + task, model='r38', task=task) for task in TASKS],
        input_manifests=manifests, parents={str(k): v for k, v in PARENTS.items()},
        architecture='Unmodified frozen R38 dense-support model; existing 148-class task head and fine labels',
        model_source=str(DENSE / 'tools/cpsf_r38_dense/model.py'),
        geometry='Unmodified dense loader after the inherited fine-to-coarse geometry adapter; original FPS, weights, fine support and Utonia input',
        training='Inherited R38 loop; exact R34 fine-task seed, split, epoch orders, loss, optimizer, budget and fixed-final evaluation; fresh initialization',
        selection='fixed final epoch development; no test feedback',
        task_accounting=dict(h01_global='reuse completed dense A1', flywire_coarse='continue live dense A1',
                             h01_local='skip retraining only after zero density triggers and R38 equivalence audit'),
        failure_policy='stop on failed acceptance or training; no automatic retry',
        resources='admit after accounting for live parent reservations; preserve existing jobs'))
    print('FROZEN', sha(ROOT / 'protocol.json'), flush=True)


if __name__ == '__main__':
    main()

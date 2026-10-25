"""Same seed, cells, budget and decoder objective for three real encoders."""
import argparse
import hashlib
import inspect
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from .common import ROOT, read, save, seed, sha, verify
from .metrics import aggregate, cell_metrics, contract
from .models import CountModel, read_item


def source_manifest(model):
    paths = set()
    for module in model.modules():
        try:
            path = inspect.getfile(type(module))
        except (TypeError, OSError):
            continue
        if '/tools/' in path or '/vendor/' in path:
            paths.add(path)
    return {path: sha(path) for path in sorted(paths)}


def training_rate(rows):
    count, exposure = 0., 0.
    for row in rows:
        assert row['split'] == 'train'
        with np.load(row['label_path']) as a:
            count += float(a['target'].sum())
            exposure += float(a['length'].sum())
    assert exposure > 0 and count > 0
    return count / exposure


def loss(expected, target):
    return (expected - target * expected.clamp_min(1e-12).log()).mean()


@torch.no_grad()
def evaluate(model, rows, name, folder=None):
    model.eval()
    result = []
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        item, y = read_item(row, 'cuda', name)
        expected = model(item)
        assert torch.isfinite(expected).all() and (expected > 0).all()
        yy, mm, ll = y.cpu().numpy(), expected.cpu().numpy(), item['length'].cpu().numpy()
        result.append(dict(root_id=row['root_id'], **cell_metrics(yy, mm, ll)))
        if folder:
            np.savez_compressed(folder / (row['root_id'] + '.npz'), target=yy, expected=mm, length=ll)
        del item, y, expected
    return aggregate(result), result


def atomic_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def verify_campaign():
    protocol = verify()
    path = ROOT / 'training_protocol.json'
    if path.exists():
        campaign = read(path)
        for source, expected in campaign['files'].items():
            assert sha(source) == expected, ('training source drift', source)
    return protocol


def accept(name, pilot):
    seed()
    limit = float(os.environ.get('MICRONS_GPU_LIMIT_MIB', '20000'))
    torch.cuda.set_per_process_memory_fraction(limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    protocol = verify_campaign()
    manifest = read(ROOT / ('pilot_manifest.json' if pilot else 'manifest.json'))
    training = [r for r in manifest['records'] if r['split'] == 'train']
    # Largest-shape acceptance uses train cells only; test stays sealed.
    extremes = {max(training, key=lambda r: r[k])['root_id'] for k in ['nodes', 'regions', 'points', 'query_regions']}
    shape_audit = []
    if not pilot:
        for row in training:
            assert sha(row['input_path']) == row['input_sha256']
            item = torch.load(row['input_path'], map_location='cpu', weights_only=False)
            g = item['r38']
            shape_audit.append(dict(root_id=row['root_id'], patches=g['patches'],
                 incidences=len(g['pr']), context_voxels=len(g['ucoord']),
                 point_voxels=len(item['point']['coord']),
                 max_raw_group=int(g.get('dense_raw_counts', g['raw_counts']).max())))
            del item, g
        for key in ['patches', 'incidences', 'context_voxels', 'point_voxels', 'max_raw_group']:
            extremes.add(max(shape_audit, key=lambda r: r[key])['root_id'])
    rows = [r for r in training if r['root_id'] in extremes]
    initial_rate = training_rate(training)
    model = CountModel(name, initial_rate).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['training']['lr'], weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    losses, gradient_prefixes = [], set()
    # Upstream Utonia registers a mask token that is unused without masking.
    # Check actual encoder updates rather than assuming its first parameter runs.
    before = {key: param.detach().cpu().clone() for key, param in model.encoder.named_parameters()}
    for row in rows:
        for _ in range(2):
            item, y = read_item(row, 'cuda', name)
            optimizer.zero_grad(set_to_none=True)
            expected = model(item)
            assert expected.shape == y.shape and torch.isfinite(expected).all()
            value = loss(expected, y)
            value.backward()
            active = []
            for key, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), ('nonfinite gradient', key)
                    if param.grad.abs().max() > 0:
                        active.append(key)
            gradient_prefixes.update(active)
            assert any(key.startswith('encoder.') for key in active)
            if name == 'r38_dense':
                assert any('cmm_pair_fusion' in key for key in active)
                assert any('region_message' in key for key in active)
                assert any('native_point' in key for key in active)
                assert any('raw_refine' in key for key in active)
            elif name == 'sgformer':
                assert any('trans_conv' in key for key in active) and any('gnn' in key for key in active)
            else:
                for level in range(5):
                    assert any(f'backbone.enc.enc{level}.' in key and 'qkv' in key for key in active)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(value.detach()))
            del item, y, expected, value
    changed = [key for key, param in model.encoder.named_parameters()
               if not torch.equal(before[key], param.detach().cpu())]
    assert changed, 'No encoder parameter changed'
    report = dict(passed=True, name=name, pilot=pilot, rows=[r['root_id'] for r in rows],
                  optimization_steps=len(losses), losses=losses, seed=42, pretrained=False,
                  metric_contract=contract(), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  active_gradient_parameters=len(gradient_prefixes), changed_encoder_parameters=len(changed),
                  training_shape_audit=shape_audit,
                  implementation_sources=source_manifest(model),
                  acceptance_weights_discarded=True)
    save(ROOT / 'acceptance' / (name + ('_pilot' if pilot else '') + '.json'), report)
    print(report, flush=True)


def main(name):
    seed()
    limit = float(os.environ.get('MICRONS_GPU_LIMIT_MIB', '20000'))
    torch.cuda.set_per_process_memory_fraction(limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    protocol = verify_campaign()
    cfg = protocol['training']
    manifest = read(ROOT / 'manifest.json')
    assert manifest['complete'] and len(manifest['records']) == 538
    assert read(ROOT / 'acceptance' / (name + '.json'))['passed']
    rows = manifest['records']
    train, val, test = [[r for r in rows if r['split'] == split] for split in ['train', 'val', 'test']]
    assert [len(train), len(val), len(test)] == [379, 79, 80]
    assert len({r['nucleus_id'] for r in rows}) == 538
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'first_update.json').exists(), 'No implicit retry/overwrite'
    model = CountModel(name, training_rate(train)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_updates = math.ceil(len(train) / cfg['effective_batch_cells']) * cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, eta_min=cfg['minimum_lr'])
    save(out / 'identity.json', dict(name=name, seed=42, pretrained=False, selection=cfg['selection'],
         protocol_sha256=sha(ROOT / 'protocol.json'), manifest_sha256=sha(ROOT / 'manifest.json'),
         training_protocol_sha256=sha(ROOT / 'training_protocol.json'),
         parameters=sum(p.numel() for p in model.parameters()), implementation_sources=source_manifest(model)))
    best, updates, start = float('inf'), 0, time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        order = np.random.default_rng(42 + epoch).permutation(len(train))
        order_hash = hashlib.sha256('\n'.join(train[i]['root_id'] for i in order).encode()).hexdigest()
        model.train()
        training_losses = []
        for at in range(0, len(order), cfg['effective_batch_cells']):
            batch = order[at:at + cfg['effective_batch_cells']]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                item, target = read_item(train[index], 'cuda', name)
                expected = model(item)
                value = loss(expected, target)
                assert torch.isfinite(value)
                (value / len(batch)).backward()
                training_losses.append(float(value.detach()))
                del item, target, expected, value
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            updates += 1
            if updates == 1:
                save(out / 'first_update.json', dict(updates=updates, epoch=epoch, pid=os.getpid(),
                     gradient_norm=float(norm), scratch=True, time=time.time()))
            save(out / 'status.json', dict(state='training', epoch=epoch, epochs=cfg['epochs'],
                 updates=updates, total_updates=total_updates, pid=os.getpid(), elapsed_seconds=time.time() - start))
        metrics, _ = evaluate(model, val, name)
        selected = metrics['cell_macro_poisson_deviance'] < best
        if selected:
            best = metrics['cell_macro_poisson_deviance']
            atomic_torch(out / 'best.pt', dict(model=model.state_dict(), epoch=epoch, updates=updates))
            save(out / 'best_selection.json', dict(epoch=epoch, updates=updates, validation=metrics,
                 selection_metric='cell_macro_poisson_deviance', test_used=False))
        atomic_torch(out / 'latest.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, updates=updates, rng=torch.get_rng_state()))
        history = dict(epoch=epoch, updates=updates, sample_order_sha256=order_hash,
                       train_nll=float(np.mean(training_losses)), validation=metrics, selected=selected)
        with (out / 'history.jsonl').open('a') as f:
            f.write(__import__('json').dumps(history) + '\n')
        print(__import__('json').dumps(history), flush=True)
    assert updates == total_updates
    selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'])
    # Test labels first enter this worker only after the fixed training budget.
    validation, vrows = evaluate(model, val, name, out / 'predictions/val')
    testing, trows = evaluate(model, test, name, out / 'predictions/test')
    save(out / 'metrics.json', dict(complete=True, selected_epoch=selected['epoch'], epochs=cfg['epochs'],
         updates=updates, validation=validation, test=testing, test_evaluated_after_training=True,
         selected_checkpoint_sha256=sha(out / 'best.pt'), validation_cells=vrows, test_cells=trows))
    save(out / 'status.json', dict(state='complete', updates=updates, epochs=cfg['epochs'], elapsed_seconds=time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--accept', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    a = parser.parse_args()
    try:
        if a.accept:
            accept(a.model, a.pilot)
        else:
            main(a.model)
    except Exception as exc:
        save(ROOT / 'failures' / (a.model + ('_pilot' if a.pilot else '_accept' if a.accept else '') + '.json'),
             dict(error_type=type(exc).__name__, error=str(exc), pid=os.getpid()))
        raise

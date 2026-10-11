"""Compare standalone code with frozen originals, including real cached input.

CPU only. --real requires the existing authorized server data paths.
Use separate processes per task to isolate historical Python import namespaces.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from model_full import TASKS, build_model, full_state
from example import synthetic_input

ROOT = Path(__file__).resolve().parent


def reference(task):
    label = 'h01_flywire' if task in TASKS[:3] else task
    sys.path.insert(0, str(ROOT / 'snapshots' / label))
    if task in TASKS[:3]:
        from tools.cpsf_fixed_core.model import Model
        return Model(task, 'full')
    if task == 'ted':
        from tools.cpsf_fixed_nredd.model import Model
        return Model(task, 'full')
    from tools.cpsf_directional.model import Model
    return Model(task, 'full', 'directional_xyz')


def real_input(task):
    info = json.loads((ROOT / 'server_export.json').read_text())
    label = 'h01_flywire' if task in TASKS[:3] else task
    server = Path(info['roots'][label]['server_root'])
    if task in TASKS[:3]:
        from tools.cpsf_fixed_core.common import rows, load_graph
        rr, _ = rows(task, 'full')
        row = min((r for r in rr if r['split'] == 'train'), key=lambda r: Path(r['path']).stat().st_size)
        g = load_graph(row, task)
    else:
        from tools.cpsf_fixed_nredd.common import load
        if task == 'ted':
            p = json.loads((server / 'protocol.json').read_text())
            path = Path(p['manifests']['ted']['joint'])
        else:
            path = server / 'data/tlp/joint/manifest.json'
        rr = json.loads(path.read_text())['records']
        row = min((r for r in rr if r['split'] == 'train'), key=lambda r: Path(r['path']).stat().st_size)
        g = load(row['path'])['g']
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
    return g, dict(key=row['key'], path=row['path'], sha256=row['sha256'], split='train')


def compare(a, b, g, label):
    result = {'label': label}
    for training in (False, True):
        a.train(training); b.train(training)
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        if training:
            x, y = a(g), b(g)
        else:
            with torch.no_grad(): x, y = a(g), b(g)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
        record = dict(output_max_abs=float((x-y).abs().max()), shape=list(x.shape))
        if training:
            x.square().mean().backward(); y.square().mean().backward()
            max_error = 0.; active = 0
            for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
                assert ka == kb and (pa.grad is None) == (pb.grad is None), (ka, kb)
                if pa.grad is not None:
                    assert torch.isfinite(pa.grad).all() and torch.isfinite(pb.grad).all()
                    torch.testing.assert_close(pa.grad, pb.grad, atol=1e-7, rtol=1e-6)
                    max_error = max(max_error, float((pa.grad-pb.grad).abs().max()))
                    active += int(bool(pa.grad.abs().max() > 0))
            record.update(gradient_max_abs=max_error, nonzero_gradient_parameters=active)
        result['train' if training else 'eval'] = record
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--real', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42); original = reference(args.task)
    torch.manual_seed(42); compact = build_model(args.task)
    sa, sb = original.state_dict(), compact.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa), 'Initialization changed'
    ck = torch.load(ROOT / 'checkpoints' / args.task / 'final.pt', map_location='cpu', weights_only=True)
    identity = json.loads((ROOT / 'checkpoints' / args.task / 'identity.json').read_text())
    state = full_state(args.task, ck, identity)
    original.load_state_dict(state, strict=True); compact.load_state_dict(state, strict=True)
    historical = None
    if args.task in ('h01_local', 'h01_global'):
        from tools.cpsf_v26_complete.model import Model as HistoricalModel
        historical = HistoricalModel(args.task, 'fixed_residual')
        historical.load_state_dict(ck['model'] if 'model' in ck else ck, strict=True)
    results = [compare(original, compact, synthetic_input(args.task), 'synthetic')]
    if args.task == 'flywire_coarse':
        results.append(compare(original, compact, synthetic_input(args.task, True), 'chunked_surface'))
    sample = None
    if args.real:
        g, sample = real_input(args.task)
        results.append(compare(original, compact, g, 'real_train_cache'))
        if historical is not None:
            historical.eval(); compact.eval()
            with torch.no_grad():
                old, new = historical(g), compact(g)
            torch.testing.assert_close(old, new, atol=1e-7, rtol=1e-6)
            results.append(dict(label='historical_fixed_residual_checkpoint',
                                output_max_abs=float((old-new).abs().max()),
                                removed_only_unused_zero_gate_tensors=True))
    report = dict(task=args.task, passed=True, device='cpu', torch=torch.__version__,
                  source_sha256=hashlib.sha256((ROOT / 'model_full.py').read_bytes()).hexdigest(),
                  initialization_exact=True, strict_checkpoint_load=True,
                  parameters=sum(x.numel() for x in compact.parameters()),
                  test_used=False, real_sample=sample, comparisons=results)
    (ROOT / 'validation').mkdir(exist_ok=True)
    (ROOT / 'validation' / (args.task + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

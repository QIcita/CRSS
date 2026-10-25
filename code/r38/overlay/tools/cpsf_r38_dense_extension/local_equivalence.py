"""Prove the dense branch is inactive on the frozen H01 local cohort."""
import gc
import os
import torch
from .common import ROOT, R38, R34, rows, read, save, sha, verify
from .backend import Backend, digest, loss_for
from tools.cpsf_cmm_pair_fusion.backend import Backend as OriginalBackend
from tools.cpsf_cmm_pair_fusion.worker import optimizer


def main():
    verify()
    records = rows('h01_local')
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'val']
    assert (len(train), len(val)) == (716, 214)
    density = {}
    for name, cohort in [('train', train), ('val', val)]:
        values = [r['points'] / r['patches'] for r in cohort]
        density[name] = dict(samples=len(cohort), minimum_mean_points=min(values),
                             regrouped_samples=sum(v < 10 for v in values))
        assert min(values) >= 10
    old = OriginalBackend('r38', 'h01_local', train)
    new = Backend('r38', 'h01_local', train)
    assert digest(old.model) == digest(new.model)
    cfg = read(R34 / 'protocol.json')['tasks']['h01_local']
    opt0, _ = optimizer(old, cfg)
    opt1, _ = optimizer(new, cfg)
    warm = min(train, key=lambda r: r['points'])
    trace = []
    for step in range(2):
        losses = []
        for backend, opt in ((old, opt0), (new, opt1)):
            torch.manual_seed(42 + step)
            backend.model.train()
            opt.zero_grad(set_to_none=True)
            g, y, _ = backend.item(warm)
            assert 'dense_size' not in g
            loss = loss_for(backend.model(g), y, 'h01_local', backend.scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.model.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        assert losses[0] == losses[1] and digest(old.model) == digest(new.model)
        trace.append(dict(step=step + 1, loss=losses[0], all_parameters_exact=True))
    del opt0, opt1, g, loss
    gc.collect()
    torch.cuda.empty_cache()
    result_dir = R38 / 'results/r38_h01_local'
    metrics = read(result_dir / 'metrics.json')
    assert sha(result_dir / 'final.pt') == metrics['final_sha256']
    state = torch.load(result_dir / 'final.pt', map_location='cpu', weights_only=False)['model']
    old.model.load_state_dict(state, strict=True)
    new.model.load_state_dict(state, strict=True)
    del state
    old.model.eval()
    new.model.eval()
    before = digest(new.model)
    cases = []
    extremes = list({r['key']: r for r in [min(train, key=lambda r: r['points']/r['patches'])]
                    + [max(train, key=lambda r: r[k]) for k in ('points', 'nodes', 'patches', 'regions', 'incidences')]}.values())
    with torch.no_grad():
        for row in extremes + val:
            g0, _, _ = old.item(row)
            g1, _, _ = new.item(row)
            assert 'dense_size' not in g1
            for key, value in g0.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, g1[key]), key
            a, b = old.eval_forward(g0), new.eval_forward(g1)
            assert torch.equal(a, b), row['key']
            cases.append(dict(key=row['key'], split=row['split'], output_exact=True))
            if len(cases) % 20 == 0:
                save(ROOT / 'local_equivalence_progress.json', dict(processed=len(cases), total=len(extremes)+len(val), pid=os.getpid()))
    assert digest(new.model) == before == digest(old.model)
    result = dict(passed=True, density=density, original_initialization_exact=True,
                  two_optimizer_steps=trace, final_checkpoint_sha256=metrics['final_sha256'],
                  final_checkpoint_strict_load=True, cases=cases, all_development_predictions_exact=True,
                  eval_state_unchanged=True, retraining_skipped=True, new_performance_row=False,
                  reason='No sample triggers dense grouping; initialization, updates and final-checkpoint forward agree exactly',
                  test_used=False, protocol_sha256=sha(ROOT / 'protocol.json'),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(ROOT / 'h01_local_equivalence.json', result)
    print('LOCAL_EQUIVALENT', len(cases), flush=True)


if __name__ == '__main__':
    main()

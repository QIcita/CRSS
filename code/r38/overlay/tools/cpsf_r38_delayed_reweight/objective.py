"""Weight the whole per-cell smoothed CE, never the single-cell mean denominator."""
from .common import ROOT, TASK, read, save
from tools.cpsf_geometry_rebench.backend import loss_for as original_loss

_EPOCH = None
_WEIGHTS = None
RECORD_STATE = False

def configure(training):
    global _WEIGHTS
    assert all(r['split'] == 'train' for r in training)
    counts = [sum(int(r['target']) == c for r in training) for c in range(6)]
    assert all(counts)
    n = sum(counts)
    u = [(n/(6*count))**.25 for count in counts]
    mean = sum(count*x for count, x in zip(counts, u))/n
    _WEIGHTS = tuple(x/mean for x in u)
    p = read(ROOT/'protocol.json')['loss']
    assert counts == p['train_counts'] and list(_WEIGHTS) == p['class_weights']
    assert abs(sum(count*x for count, x in zip(counts, _WEIGHTS))/n-1) < 1e-12
    set_epoch(1)
    return counts, _WEIGHTS

def ramp(epoch):
    assert isinstance(epoch, int) and 1 <= epoch <= 50
    return 0. if epoch <= 30 else min((epoch-30)/5., 1.)

def weights(epoch):
    assert _WEIGHTS is not None
    a = ramp(epoch)
    return tuple(1+a*(w-1) for w in _WEIGHTS)

def set_epoch(epoch):
    global _EPOCH
    ramp(epoch); _EPOCH = epoch
    if RECORD_STATE:
        save(ROOT/'results/r38_flywire_coarse/objective_state.json', dict(
            epoch=epoch, ramp=ramp(epoch), weights=weights(epoch),
            label_smoothing=.05, reduction='target_weight * per_cell_CE / group_cell_count',
            primary_metric='macro_f1'))

def loss_for(z, y, task, scale):
    assert task == TASK and _EPOCH is not None
    base = original_loss(z, y, task, scale)
    if _EPOCH <= 30:
        # Same tensor expression and autograd graph as the parent until epoch 30.
        return base
    return base * weights(_EPOCH)[int(y)]

"""Equal-cell metrics; a cell's regions never count as independent cells."""
import numpy as np


def cell_metrics(target, expected, length):
    y, mu, length = [np.asarray(x, dtype=np.float64) for x in (target, expected, length)]
    assert y.shape == mu.shape == length.shape and len(y)
    assert np.isfinite(mu).all() and (mu > 0).all() and (length > 0).all()
    assert (y >= 0).all() and np.allclose(y, np.rint(y))
    term = mu - y
    positive = y > 0
    term[positive] += y[positive] * np.log(y[positive] / mu[positive])
    out = dict(poisson_deviance=float(np.maximum(2 * term, 0).mean()),
               region_density_mae=float(np.abs(y / length - mu / length).mean()),
               total_count_mae=float(abs(y.sum() - mu.sum())), regions=len(y),
               true_total=float(y.sum()), predicted_total=float(mu.sum()))
    if y.sum() > 0:
        p, q = y / y.sum(), mu / mu.sum()
        m = (p + q) / 2
        klp = np.sum(p[positive] * np.log2(p[positive] / m[positive]))
        klq = np.sum(q * np.log2(q / m))
        out['js_divergence_base2'] = float(max(0, (klp + klq) / 2))
    else:
        out['js_divergence_base2'] = None
    return out


def aggregate(rows):
    assert rows
    out = {'cells': len(rows), 'regions': sum(r['regions'] for r in rows)}
    for key in ['poisson_deviance', 'region_density_mae', 'total_count_mae', 'js_divergence_base2']:
        values = [r[key] for r in rows if r[key] is not None]
        out['cell_macro_' + key] = float(np.mean(values)) if values else None
    out['js_eligible_cells'] = sum(r['js_divergence_base2'] is not None for r in rows)
    return out


def contract():
    a = cell_metrics([20, 80], [20, 80], [2, 2])
    b = cell_metrics([20, 80], [80, 20], [2, 2])
    z = cell_metrics([0, 0], [1, 1], [1, 1])
    assert a['poisson_deviance'] == 0 and a['js_divergence_base2'] == 0
    assert b['total_count_mae'] == 0 and b['poisson_deviance'] > 0 and b['js_divergence_base2'] > 0
    assert z['poisson_deviance'] == 2 and z['js_divergence_base2'] is None
    average = aggregate([a, z])
    assert average['cell_macro_poisson_deviance'] == 1 and average['js_eligible_cells'] == 1
    return {'exact_prediction': True, 'redistribution_detected': True, 'zero_counts': True, 'equal_cell_weight': True}

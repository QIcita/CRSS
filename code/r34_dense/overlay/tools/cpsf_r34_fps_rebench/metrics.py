"""Fixed 148-class macro metrics, including classes with zero predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from .common import ROOT, read


def classification(y, z):
    y = np.asarray(y, dtype=int)
    z = np.asarray(z)
    assert z.shape == (len(y), 148) and np.isfinite(z).all()
    pred = z.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(148), zero_division=0)
    labels = read(ROOT / 'labels.json')
    counts = np.array(labels['total_support'])
    grouped = {}
    for name, mask in [('head', counts >= 100), ('middle', (counts >= 20) & (counts < 100)),
                       ('tail', counts < 20)]:
        grouped[name] = dict(classes=int(mask.sum()), development_samples=int(support[mask].sum()),
                             macro_f1=float(f1[mask].mean()))
    return dict(macro_f1=float(f1.mean()), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(recall.mean()),
                top5_accuracy=float(np.any(np.argsort(-z, axis=1, kind='stable')[:, :5] == y[:, None], axis=1).mean()),
                confusion_matrix=confusion_matrix(y, pred, labels=np.arange(148)).tolist(),
                per_class=[dict(label=i, name=labels['fine'][i], precision=float(precision[i]),
                                recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])) for i in range(148)],
                frequency_groups=grouped, frequency_group_basis='frozen original cohort support, 100+/20-99/5-19')

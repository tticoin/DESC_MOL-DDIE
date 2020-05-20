import os
import sys
import numpy as np
import random
from metrics_ddie import ddie_compute_metrics
from statsmodels.stats import contingency_tables

_, output_dir1, output_dir2 = sys.argv

preds1 = np.load(os.path.join(output_dir1, 'preds.npy'))
preds2 = np.load(os.path.join(output_dir2, 'preds.npy'))

labels = np.load(os.path.join(output_dir1, 'labels.npy'))

preds1 = np.argmax(preds1, axis=1)
preds2 = np.argmax(preds2, axis=1)

result1 = ddie_compute_metrics('ddie', preds1, labels, every_type=True)
result2 = ddie_compute_metrics('ddie', preds2, labels, every_type=True)
print(result1, result2)

apbp, apbn, anbp, anbn = 0, 0, 0, 0
for p1, p2, label in zip(preds1, preds2, labels):
    if p1 == label:
        if p2 == label:
            apbp += 1
        else:
            apbn += 1
    else:
        if p2 == label:
            anbp += 1
        else:
            anbn += 1

ar = np.array([[apbp, anbp], [apbn, anbn]])
print(ar)
p = contingency_tables.mcnemar(ar).pvalue
print(p)

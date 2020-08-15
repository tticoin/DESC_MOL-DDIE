import os
import sys
import numpy as np
from metrics_ddie import ddie_compute_metrics
from scipy.special import softmax
import glob

_, outputs_dir = sys.argv

cnt = 0
for path in glob.glob(os.path.join(outputs_dir, '*')):
    labels = np.load(os.path.join(path, 'labels.npy'))
    preds = np.load(os.path.join(path, 'preds.npy'))
    result = ddie_compute_metrics('ddie', np.argmax(preds, axis=1), labels, every_type=False)
    #preds = softmax(preds, axis=1)
    print(result)
    if cnt == 0:
        ensembled_preds = preds
    else:
        ensembled_preds += preds
    cnt += 1

print('Ensemble of {} models'.format(cnt))
ensembled_result = ddie_compute_metrics('ddie', np.argmax(ensembled_preds, axis=1), labels, every_type=False)
print(ensembled_result)

import sys
import os
import pickle
import numpy as np
from metrics_ddie import ddie_compute_metrics
from scipy.special import softmax

from transformers import BertTokenizer

_, cv_dir, tsv_dir, k = sys.argv
k = int(k)

tokenizer = BertTokenizer.from_pretrained('/mnt/model/scibert_scivocab_uncased', do_lower_case=True)

sentence_lengths = [[] for i in range(k)]
for i in range(k):
    with open(os.path.join(tsv_dir, str(i+1), 'tsv', 'dev.tsv'), 'r') as f:
        lines = f.read().strip().split('\n')
    length_list = []
    for idx, line in enumerate(lines):
        sent = line.split('\t')[0]
        tokenized_sent = tokenizer.tokenize(sent)
        sentence_length = len(tokenized_sent)
        sentence_lengths[i].append(sentence_length)
"""
with open('sentence_lengths', 'wb') as f:
    pickle.dump(sentence_lengths, f)
with open('sentence_lengths', 'rb') as f:
    sentence_lengths = pickle.load(f)
"""

interval = 20
#N = 128 // interval + 1
N = 100 // interval + 1
indices = [[[] for i in range(N)] for j in range(k)]
for i in range(k):
    for idx, length in enumerate(sentence_lengths[i]):
        #if length > 128:
        #    div = 128 // interval
        if length > 100:
            div = 100 // interval
        else:
            div = length // interval
        indices[i][div].append(idx)

cnt = {}
for x in sentence_lengths:
    for xx in x:
        key = xx // interval * interval
        if key in cnt:
            cnt[key] += 1
        else:
            cnt[key] = 1
print(cnt)


#for x in indices:
#    for i,xx in enumerate(x):
#        print(i,len(xx))

#paths = ['cls', 'cnn', 'rad0', 'rad1', 'rad2', 'desc']
paths = ['cnn', 'rad1', 'desc']

for path in paths:
    print(path)

    fscores = []
    for i in range(k):
        result_path = os.path.join(cv_dir, str(i+1), path, 'eval_results.txt')
        with open(result_path, 'r') as f:
            fscore = f.read().strip().split('\n')[2].split()[-1]
            print(i+1, fscore)
            fscore = float(fscore)
            fscores.append(fscore)

    print(sum(fscores) / len(fscores))

for model_type in ('cnn', 'rad1', 'desc', 'ensemble'):
    sentence_fscores = [[] for i in range(N)]
    for j in range(N):
        micro_preds = None
        micro_labels = None
        for i in range(k):
            if len(indices[i][j]) == 0:
                continue

            if model_type != 'ensemble':
                preds_path = os.path.join(cv_dir, str(i+1), model_type, 'preds.npy')
                labels_path = os.path.join(cv_dir, str(i+1), model_type, 'labels.npy')
                preds = np.load(preds_path)
                labels = np.load(labels_path)
            else:
                rad_preds_path = os.path.join(cv_dir, str(i+1), 'rad1', 'preds.npy')
                desc_preds_path = os.path.join(cv_dir, str(i+1), 'desc', 'preds.npy')
                rad_preds = np.load(rad_preds_path)
                desc_preds = np.load(desc_preds_path)
                preds = rad_preds + desc_preds

                labels_path = os.path.join(cv_dir, str(i+1), 'cnn', 'labels.npy')
                labels = np.load(labels_path)
        
            div_preds = preds[np.array(indices[i][j])]
            div_labels = labels[np.array(indices[i][j])]
            div_result = ddie_compute_metrics('ddie', np.argmax(div_preds, axis=1), div_labels, every_type=False)
            div_fscore = div_result['microF']
            #print(j, div_fscore, len(indices[i][j]))
            sentence_fscores[j].append(div_fscore)

            if micro_preds is None:
                micro_preds = div_preds
                micro_labels = div_labels
            else:
                micro_preds = np.concatenate((micro_preds, div_preds), axis=0)
                micro_labels = np.concatenate((micro_labels, div_labels), axis=0)
        micro_result = ddie_compute_metrics('ddie', np.argmax(micro_preds, axis=1), micro_labels, every_type=False)
        #print('micro', model_type, j, micro_result['microF'])
        
    for x in sentence_fscores:
        print('macro', model_type, sentence_fscores.index(x), sum(x) / len(x))

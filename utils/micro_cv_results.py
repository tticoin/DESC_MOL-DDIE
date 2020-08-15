import sys
import os
import pickle
import numpy as np
from metrics_ddie import ddie_compute_metrics
from scipy.special import softmax

from transformers import BertTokenizer

_, cv_dir, k = sys.argv
k = int(k)

tokenizer = BertTokenizer.from_pretrained('/mnt/model/scibert_scivocab_uncased', do_lower_case=True)

"""
sentence_lengths = [[] for i in range(k)]
for i in range(k):
    with open(os.path.join('/mnt/analysis/cv'+str(k), str(i+1), 'tsv', 'dev.tsv'), 'r') as f:
        lines = f.read().strip().split('\n')
    length_list = []
    for idx, line in enumerate(lines):
        sent = line.split('\t')[0]
        tokenized_sent = tokenizer.tokenize(sent)
        sentence_length = len(tokenized_sent)
        sentence_lengths[i].append(sentence_length)
with open('sentence_lengths', 'wb') as f:
    pickle.dump(sentence_lengths, f)
"""

with open('sentence_lengths', 'rb') as f:
    sentence_lengths = pickle.load(f)

interval = 20
N = 128 // interval + 1
indices = [[[] for i in range(N)] for j in range(k)]
for i in range(k):
    for idx, length in enumerate(sentence_lengths[i]):
        if length > 128:
            div = 128 // interval
        else:
            div = length // interval
        indices[i][div].append(idx)

for x in indices:
    for i,xx in enumerate(x):
        print(i,len(xx))

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

# Ensemble
ensembled_fscores = []
sentence_fscores = [[] for i in range(N)]

for j in range(N):
    for i in range(k):
        if len(indices[i][j]) == 0:
            continue

        cnn_preds_path = os.path.join(cv_dir, str(i+1), 'cnn', 'preds.npy')
        rad_preds_path = os.path.join(cv_dir, str(i+1), 'rad1', 'preds.npy')
        desc_preds_path = os.path.join(cv_dir, str(i+1), 'desc', 'preds.npy')

        cnn_labels_path = os.path.join(cv_dir, str(i+1), 'cnn', 'labels.npy')

        cnn_preds = np.load(cnn_preds_path)
        rad_preds = np.load(rad_preds_path)
        desc_preds = np.load(desc_preds_path)
        ensembled_preds = rad_preds + desc_preds

        labels = np.load(cnn_labels_path)

        #cnn_result = ddie_compute_metrics('ddie', np.argmax(cnn_preds, axis=1), labels, every_type=False)
        #print(cnn_result)
        #ensembled_result = ddie_compute_metrics('ddie', np.argmax(ensembled_preds, axis=1), labels, every_type=False)
        #print(ensembled_result)
        #fscore = ensembled_result['microF']
        #ensembled_fscores.append(fscore)
    
        #div_preds = cnn_preds[np.array(indices[i][j])]
        #div_preds = rad_preds[np.array(indices[i][j])]
        div_preds = desc_preds[np.array(indices[i][j])]
        #div_preds = ensembled_preds[np.array(indices[i][j])]
        div_labels = labels[np.array(indices[i][j])]
        div_result = ddie_compute_metrics('ddie', np.argmax(div_preds, axis=1), div_labels, every_type=False)
        div_fsocre = div_result['microF']
        #print(j, div_result)
        sentence_fscores[j].append(div_fsocre)

        if j == 0:
            micro_preds = div_preds
            micro_labels = div_labels
        else:
            micro_preds = np.concatenate((micro_preds, div_preds), axis=0)
            micro_labels = np.concatenate((micro_labels, div_labels), axis=0)
    micro_result = ddie_compute_metrics('ddie', np.argmax(micro_preds, axis=1), micro_labels, every_type=False)
    print('micro', j, micro_result['microF'])
        
for x in sentence_fscores:
    print(sum(x) / len(x))
#print(sum(ensembled_fscores) / len(ensembled_fscores))

import sys
import os
import pickle
import statistics
import numpy as np
from metrics_ddie import ddie_compute_metrics
from scipy.special import softmax

from transformers import BertTokenizer

_, tsv_dir, cv_dir, k = sys.argv
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
model_list = ('cnn', 'desc', 'rad1', 'ensemble')


# Ensemble
ensembled_fscores = []
sentence_fscores = [[] for i in range(N)]

for model_name in model_list:
    print(model_name)
    for i in range(k):
        if model_name == 'ensemble':
            rad_preds = np.load(os.path.join(cv_dir, str(i+1), 'rad1', 'preds.npy'))
            desc_preds = np.load(os.path.join(cv_dir, str(i+1), 'desc', 'preds.npy'))
            preds = rad_preds + desc_preds
            labels = np.load(os.path.join(cv_dir, str(i+1), 'rad1', 'labels.npy'))
        else:
            preds= np.load(os.path.join(cv_dir, str(i+1), model_name, 'preds.npy'))
            labels= np.load(os.path.join(cv_dir, str(i+1), model_name, 'labels.npy'))

        for j in range(N):
            if len(indices[i][j]) == 0:
                continue

        
            div_preds = preds[np.array(indices[i][j])]
            div_labels = labels[np.array(indices[i][j])]
            div_result = ddie_compute_metrics('ddie', np.argmax(div_preds, axis=1), div_labels, every_type=False)
            div_fscore = div_result['microF']
            #print(j, div_fscore)
            sentence_fscores[j].append(div_fscore)
            
    for x in sentence_fscores:
        #print(statistics.mean(x), '\t', statistics.stdev(x))
        print(sum(x) / len(x), '\t', statistics.stdev(x))

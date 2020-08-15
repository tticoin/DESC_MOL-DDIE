import sys
import os
import pickle
import numpy as np
from metrics_ddie import ddie_compute_metrics
from scipy.special import softmax

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/mnt/model/scibert_scivocab_uncased', do_lower_case=True)

with open('/mnt/corpus/test_tsv/dev.tsv', 'r') as f:
    lines = f.read().strip().split('\n')
length_list = []
for idx, line in enumerate(lines):
    sent = line.split('\t')[0]
    tokenized_sent = tokenizer.tokenize(sent)
    sentence_length = len(tokenized_sent)
    length_list.append(sentence_length)

interval = 20
N = 128 // interval + 1
indices = [[] for i in range(N)]
for idx, length in enumerate(length_list):
    if length > 128:
        div = 128 // interval
    else:
        div = length // interval
    indices[div].append(idx)

for i,x in enumerate(indices):
    print(i,len(x))

#paths = ['cls', 'cnn', 'rad0', 'rad1', 'rad2', 'desc']
paths = ['cnn', 'rad1', 'desc']


# Ensemble
cnn_preds = np.load('/mnt/output/ensembled_/cnn/preds.npy')
rad_preds = np.load('/mnt/output/ensembled_/dwin3rad0rad1rad2/sum_rad1_hidden5_output1_middle0/preds.npy')
desc_preds = np.load('/mnt/output/ensembled_/dwin3rad0rad1rad2/dwin3_size20_middle0/preds.npy')
ensembled_preds = rad_preds + desc_preds
labels = np.load('/mnt/output/ensembled_/cnn/labels.npy')

#preds = cnn_preds
#preds = rad_preds
#preds = desc_preds
preds = ensembled_preds

result = ddie_compute_metrics('ddie', np.argmax(preds, axis=1), labels, every_type=False)
print(result)

for j in range(N):
    if len(indices[j]) == 0:
        print('---')
        continue
    div_preds = preds[np.array(indices[j])]
    div_labels = labels[np.array(indices[j])]
    div_result = ddie_compute_metrics('ddie', np.argmax(div_preds, axis=1), div_labels, every_type=False)
    div_fscore = div_result['microF']
    print(j, div_fscore)

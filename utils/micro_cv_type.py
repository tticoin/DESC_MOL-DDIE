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


paths = ('cnn', 'rad1', 'desc')
#paths = ('cnn', 'desc')

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

#label_list = ('microF', 'Mechanism_F', 'Effect_F', 'Advise_F', 'Int._F')
label_list = ('Mechanism_F', 'Effect_F', 'Advise_F', 'Int._F')
model_list = ('cnn', 'rad1', 'desc', 'ensemble')
print_d = {'cnn': 'Text-only', 'rad1':'+ Mol (radius=1)', 'desc':'+ Desc', 'ensemble':'+ Desc + Mol (radius=1)'}
#model_list = ('cnn', 'desc')

def print_result(result_table):
    for i_, x in enumerate(result_table):
        print('& {} '.format(print_d[model_list[i_]]), end='')    

        for j_, y in enumerate(x):
            if i_ == np.argmax(result_table[:, j_]):
                print('& \\textbf{{{:.2f}}}'.format(y * 100), end=' ')
            elif i_ != 0 and y < result_table[0,:][j_]:
                print('& \\underline{{{:.2f}}}'.format(y * 100), end=' ')
            else:
                print('& {:.2f}'.format(y * 100), end=' ')
        if i_ == len(model_list)-1:
            print('\\\\\\hline')
        else:
            print('\\\\')

macro_result_dict = {}
micro_result_table = np.zeros((len(model_list), len(label_list)))

for model_i, model_name in enumerate(model_list):
    
    micro_preds = None
    micro_labels = None

    for i in range(k):
        if model_name == 'ensemble':
            rad_preds = np.load(os.path.join(cv_dir, str(i+1), 'rad1', 'preds.npy'))
            desc_preds = np.load(os.path.join(cv_dir, str(i+1), 'desc', 'preds.npy'))
            preds = rad_preds + desc_preds
            labels = np.load(os.path.join(cv_dir, str(i+1), 'rad1', 'labels.npy'))
        else:
            preds= np.load(os.path.join(cv_dir, str(i+1), model_name, 'preds.npy'))
            labels= np.load(os.path.join(cv_dir, str(i+1), model_name, 'labels.npy'))

        result = ddie_compute_metrics('ddie', np.argmax(preds, axis=1), labels, every_type=True)
        for label_type in label_list:
            key = model_list[model_i] + ' ' + label_type
            #print(i, key, result[label_type])
            #print('& {:.2f}'.format(result[label_type] * 100), end=' ')
            if key in macro_result_dict:
                macro_result_dict[key].append(result[label_type])
            else:
                macro_result_dict[key] = [result[label_type]]

        if micro_preds is None:
            micro_preds = preds
            micro_labels = labels
        else:
            micro_preds = np.append(micro_preds, preds, axis=0)
            micro_labels = np.append(micro_labels, labels, axis=0)
    micro_result = ddie_compute_metrics('ddie', np.argmax(micro_preds, axis=1), micro_labels, every_type=True)
    #for label_type in label_list:
    for label_i, label_type in enumerate(label_list):
        micro_result_table[model_i, label_i] = micro_result[label_type]
        print('MICRO', model_name, label_type, micro_result[label_type])

print('Average (MICRO) ', end='')    
print_result(micro_result_table)

#for key, value in macro_result_dict.items():
#    assert len(value) == k
#    print('MACRO', key, sum(value) / len(value))

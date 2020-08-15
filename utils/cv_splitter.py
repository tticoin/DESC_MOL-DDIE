import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import shutil


_, tsv_dir, rad0_dir, rad1_dir, rad2_dir, k, output_dir = sys.argv

with open(os.path.join(tsv_dir, 'train.tsv'), 'r') as f:
    lines = f.read().strip().split('\n')
label_d = {'negative':0, 'mechanism':1, 'effect':2, 'advise':3, 'int':4}
labels = [label_d[l.split('\t')[1]] for l in lines]

rad0 = np.load(os.path.join(rad0_dir, 'corpus_train.npy'), allow_pickle=True)
#rad1 = np.load(os.path.join(rad1_dir, 'corpus_train.npy'), allow_pickle=True)
#rad2 = np.load(os.path.join(rad2_dir, 'corpus_train.npy'), allow_pickle=True)

k = int(k)
skf = StratifiedKFold(n_splits=k)

cnt = 1
for train_idx, test_idx in skf.split(lines, labels):
    rad0[train_idx]
    #print(labels[train_idx])

    if not os.path.exists(os.path.join(output_dir, str(cnt), 'tsv')):
        os.makedirs(os.path.join(output_dir, str(cnt), 'tsv'))
    tr_tsv_path = os.path.join(output_dir, str(cnt), 'tsv', 'train.tsv')
    f_tr = open(tr_tsv_path, 'w')
    for idx in train_idx:
        f_tr.write(lines[idx]+'\n')
    f_tr.close()

    te_tsv_path = os.path.join(output_dir, str(cnt), 'tsv', 'dev.tsv')
    f_te = open(te_tsv_path, 'w')
    for idx in test_idx:
        f_te.write(lines[idx]+'\n')
    f_te.close()

    for rad_dir, rad_name in zip((rad0_dir, rad1_dir, rad2_dir), ('rad0', 'rad1', 'rad2')):
        rad_npy = np.load(os.path.join(rad_dir, 'corpus_train.npy'), allow_pickle=True)

        if not os.path.exists(os.path.join(output_dir, str(cnt), rad_name)):
            os.makedirs(os.path.join(output_dir, str(cnt), rad_name))
        train_rad = rad_npy[train_idx]
        test_rad = rad_npy[test_idx]

        np.save(os.path.join(output_dir, str(cnt), rad_name, 'corpus_train.npy'), train_rad)
        np.save(os.path.join(output_dir, str(cnt), rad_name, 'corpus_dev.npy'), test_rad)

        shutil.copy(os.path.join(rad_dir, 'config.json'), os.path.join(output_dir, str(cnt), rad_name))

    cnt += 1

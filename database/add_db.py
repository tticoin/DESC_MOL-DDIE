import pickle
import sys

def add_descsmiles(tsv, mention_dict, desc_dict, smiles_dict):
    new_tsv = '' 
    for l in tsv:
        tabs = l.split('\t')
        dbid1 = mention_dict[tabs[2].lower()]
        dbid2 = mention_dict[tabs[3].lower()]

        desc1 = desc_dict.get(dbid1, '')
        desc2 = desc_dict.get(dbid2, '')

        desc1 = desc1.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        desc2 = desc2.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

        smiles1 = smiles_dict.get(dbid1, '')
        smiles2 = smiles_dict.get(dbid2, '')

        new_tsv += l + '\t' + desc1 + '\t' + desc2 + '\t' + smiles1 + '\t' + smiles2 + '\n'

    return new_tsv

tsv_in, mention_dict_in, smiles_dict_in, desc_dict_in, new_tsv_out = sys.argv[1:]
with open(tsv_in, 'r') as f:
    tsv = f.read().strip().split('\n')
with open(mention_dict_in, 'rb') as f:
    mention_dict = pickle.load(f)
with open(desc_dict_in, 'rb') as f:
    desc_dict = pickle.load(f)
with open(smiles_dict_in, 'rb') as f:
    smiles_dict = pickle.load(f)

new_tsv = add_descsmiles(tsv, mention_dict, desc_dict, smiles_dict)
with open(new_tsv_out, 'w') as f:
    f.write(new_tsv)

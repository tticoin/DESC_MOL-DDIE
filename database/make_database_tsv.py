import sys
import os
import pickle as pkl
import itertools
from lxml import etree


drugbank_xml, smiles_dict_path, desc_dict_path, test_tsv_path, mention_test_dict_path, output_tsv_dir = sys.argv[1:]

root = etree.parse(drugbank_xml, parser=etree.XMLParser())

with open(smiles_dict_path, 'rb') as f:
    smiles_dict = pkl.load(f)
with open(desc_dict_path, 'rb') as f:
    desc_dict = pkl.load(f)

all_drugs = set()
interacted_pairs = set()

# Get interacted drug pairs
for drug in root.xpath('./*[local-name()="drug"]'):
    drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text
    if drug_id not in smiles_dict or drug_id not in desc_dict:
        continue
    all_drugs.add(drug_id)

    interacted_drugs = drug.xpath('./*[local-name()="drug-interactions"]')[0]
    for interacted_drug in interacted_drugs:
        interacted_drug_id = interacted_drug.xpath('./*[local-name()="drugbank-id"]')[0].text
        if interacted_drug_id not in smiles_dict or interacted_drug_id not in desc_dict:
            continue
        pair = sorted([drug_id, interacted_drug_id])
        interacted_pairs.add(':'.join(pair))

print(len(interacted_pairs))
# Exclude drug pairs wchich appear in test corpus
with open(mention_test_dict_path, 'rb') as f:
    mention_test_dict= pkl.load(f)
with open(test_tsv_path, 'r') as f:
    test_tsv = f.read().strip().split('\n')
for line in test_tsv:
    tabs = line.split('\t')
    mention1 = tabs[2].lower()
    mention2 = tabs[3].lower()
    dbid1 = mention_test_dict[mention1]
    dbid2 = mention_test_dict[mention2]
    if dbid1 is not None and dbid2 is not None:
        pair = ':'.join(sorted([dbid1, dbid2]))
        if pair in interacted_pairs:
            interacted_pairs.remove(pair)
print(len(interacted_pairs))


# Create pseudo negative pairs
all_pairs = set()
for x in itertools.combinations(all_drugs, 2):
    all_pairs.add(':'.join(sorted(list(x))))
not_interacted_pairs = all_pairs - interacted_pairs

interacted_pairs = list(interacted_pairs)
not_interacted_pairs = list(not_interacted_pairs)
n_pos = len(interacted_pairs)
n_neg = len(not_interacted_pairs)
n = min(n_pos, n_neg)
ratio = 4
train_out = open(os.path.join(output_tsv_dir, 'train.tsv'), 'w')
dev_out = open(os.path.join(output_tsv_dir, 'dev.tsv'), 'w')
for i in range(n_pos):
    for j in (0, 1):
        pairs = (not_interacted_pairs, interacted_pairs)[j]
        id1, id2 = pairs[i].split(':')
        smiles1 = smiles_dict[id1]
        smiles2 = smiles_dict[id2]
        desc1 = desc_dict[id1]
        desc2 = desc_dict[id2]
        label = ('negative', 'positive')[j]
        train_out.write('NoSent\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(label, id1, id2, desc1, desc2, smiles1, smiles2))
train_out.close()
dev_out.close()

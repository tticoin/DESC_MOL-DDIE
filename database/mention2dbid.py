import sys
import pickle
from lxml import etree
from multiprocessing import Pool


# Convert corpus mention to drugbank id

def relaxed_match(mention, dict_, key='all'):
    matched = {} # key: drugbank_id, value: matching rate
    for k, v in dict_.items():
        if key == 'all':
            if mention.lower() in k.lower() or k.lower() in mention.lower():
                matched[v] = abs(len(k)-len(mention))
        elif key == 'mention':
            if mention.lower() in k.lower():
                matched[v] = abs(len(k)-len(mention))
        elif key == 'dict':
            if k.lower() in mention.lower():
                matched[v] = abs(len(k)-len(mention))

    if len(matched) == 0:
        return None
    else:
        return min(matched, key=matched.get)

def exact_match(mention, dict_):
    for k, v in dict_.items():
        if k.lower() == mention.lower():
            return v
    return None

def wrapper_e(args):
    return exact_match(*args)

def wrapper_r(args):
    return relaxed_match(*args)


#_, base, name_dict_in = sys.argv
_, tsv_in, entry_dict_in, mention_dict_out = sys.argv

with open(tsv_in, 'r') as f:
    mentions = f.read().strip().split('\n')

mention1_instances = [x.split('\t')[2] for x in mentions]
mention2_instances = [x.split('\t')[3] for x in mentions]

mentions = set()
for m1, m2 in zip(mention1_instances, mention2_instances):
    mentions.add(m1.lower())
    mentions.add(m2.lower())
mentions = list(mentions)

with open(entry_dict_in, 'rb') as f:
    entry_dict = pickle.load(f)
#name_dict, brand_dict, product_dict, syn_dict = name_dicts
name_dict, brand_dict, product_dict, atc_dict, synonym_dict = entry_dict

atc_dict_ = {}
for k,v in atc_dict.items():
    if len(v) == 1:
    #if True:
        atc_dict_[k] = v[0]

entry_dict_wo_synonym = {}
#for dict_ in (name_dict, brand_dict, product_dict):
for dict_ in (name_dict, brand_dict, product_dict, atc_dict_):
    entry_dict_wo_synonym.update(dict_)

p = Pool()
args = [(m, entry_dict_wo_synonym) for m in mentions]
args_ = [(m, synonym_dict, 'mention') for m in mentions]
res = p.map(wrapper_r, args)
res_ = p.map(wrapper_r, args_)
merged_res = [r_ if r is None else r for r,r_ in zip(res, res_)]

print(len(res))
print(res.count(None))

print(len(merged_res))
print(merged_res.count(None))

mention_dict = {}
for m, dbid in zip(mentions, merged_res):
    mention_dict[m] = dbid

with open(mention_dict_out, 'wb') as f:
    pickle.dump(mention_dict, f)

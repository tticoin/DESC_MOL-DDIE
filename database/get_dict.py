import sys
from lxml import etree
import pickle
from rdkit import Chem

def dump_dict(dict_, path):
    with open(path, 'wb') as f:
        pickle.dump(dict_, f)

def get_smiles_dict(root):
    smiles_dict = {} # key:drugbank_id, value:smiles

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text

        for kind in drug.xpath('.//*[local-name()="kind"]'):
            if kind.text == 'SMILES':
                smiles = kind.getparent()[1].text
                # Confirm the validity of smiles
                try:
                    mol = Chem.MolFromSmiles(smiles)
                except:
                    break
                smiles_dict[drug_id] = smiles
    return smiles_dict

def get_desc_dict(root):
    desc_dict = {}

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text

        desc_text = drug.xpath('./*[local-name()="description"]')[0].text
        if desc_text is not None:
            desc_text = desc_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            desc_dict[drug_id] = desc_text
    return desc_dict

def get_name_dict(root):
    name_dict = {} # key:name, value:drugbank_id
    brand_dict = {}
    product_dict = {}
    syn_dict = {}
    #mixture_dict = {}
    #direct_parent_dict = {}
    atc_dict = {}

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text

        #if drug_id not in smiles_dict:
        #    continue

        # Name
        name_text = drug.xpath('./*[local-name()="name"]')[0].text
        name_dict[name_text] = drug_id
        # Brand
        for brand in drug.xpath('./*[local-name()="international-brands"]')[0]:
            brand_text = brand.xpath('./*[local-name()="name"]')[0].text
            name_dict[brand_text] = drug_id
        # Product
        for product in drug.xpath('./*[local-name()="products"]')[0]:
            product_text = product.xpath('./*[local-name()="name"]')[0].text
            name_dict[product_text] = drug_id
        # Synonyms
        for syn in drug.xpath('./*[local-name()="synonyms"]')[0]:
            syn_text = syn.text
            syn_dict[syn_text] = drug_id

        # ATC-code
        for atcs in drug.xpath('./*[local-name()="atc-codes"]')[0]:
            for atc in atcs:
                atc_text = atc.text
                if atc_text in atc_dict:
                    ids = atc_dict[atc_text]
                    ids.append(drug_id)
                    atc_dict[atc_text] = ids
                else:
                    atc_dict[atc_text] = [drug_id]

    return (name_dict, brand_dict, product_dict, atc_dict, syn_dict)

_, xml_path, name_dict_out, desc_dict_out, smiles_dict_out = sys.argv

root = etree.parse(xml_path, parser=etree.XMLParser())
name_dict = get_name_dict(root)
desc_dict = get_desc_dict(root)
smiles_dict = get_smiles_dict(root)

dump_dict(name_dict, name_dict_out)
dump_dict(desc_dict, desc_dict_out)
dump_dict(smiles_dict, smiles_dict_out)


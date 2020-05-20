import os
import sys
import json

from collections import defaultdict

import numpy as np

from rdkit import Chem

import torch


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict, mode):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        for a in atoms:
            if mode == 'database':
                if a in fingerprint_dict:
                    return 1
                else:
                    return 0
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                if mode == 'database':
                    if fingerprint in fingerprint_dict:
                        return 1
                    else:
                        return 0
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(corpus_tsv_dir, database_tsv_dir, radius):

    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filepath, mode):

        """Load a dataset."""
        with open(filepath, 'r') as f:
            #smiles_property = f.readline().strip().split()
            #data_original = f.read().strip().split('\n')
            data_original = f.readlines()
        print(len(data_original))

        data_original = [[data.strip('\n').split('\t')[6], data.strip('\n').split('\t')[7]]
                         for data in data_original]
        """Exclude the data contains '.' in its smiles.
        data_original = [data for data in data_original
                         if '.' not in data.split()[0]]
        """

        dataset = []
        mask = []

        for data in data_original:
            dataset_ = []
            for smiles in data:
                """Replace the smiles its contains '.' with 'CC'
                   Replace the no smiles data with 'CC'"""
                if '.' in smiles or smiles == '':
                    smiles = 'CC'
                    mask = [0]
                else:
                    try:
                        Chem.AddHs(Chem.MolFromSmiles(smiles))
                        mask = [1]
                    except:
                        """Replace invalid smiles with 'CC'"""
                        smiles = 'CC'
                        mask = [0]

                """Create each data with the above defined functions."""
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict, mode)
                if mode == 'database':
                    dataset_.append(fingerprints * mask[0])
            if mode == 'database':
                dataset.append(dataset_[0]*dataset_[1])
        return dataset

    create_dataset(os.path.join(corpus_tsv_dir, 'train.tsv'), mode='corpus')
    create_dataset(os.path.join(corpus_tsv_dir, 'dev.tsv'), mode='corpus')

    database_tsv_mask = create_dataset(os.path.join(database_tsv_dir, 'all.tsv'), mode='database')

    return database_tsv_mask

corpus_tsv_dir, database_tsv_dir = sys.argv[1:]
mask0 = create_datasets(corpus_tsv_dir, database_tsv_dir, 0)
mask1 = create_datasets(corpus_tsv_dir, database_tsv_dir, 1)
mask2 = create_datasets(corpus_tsv_dir, database_tsv_dir, 2)

all_mask = [a*b*c for a,b,c in zip(mask0, mask1, mask2)]

with open(os.path.join(database_tsv_dir, 'all.tsv'), 'r') as f:
    tsv_lines = f.readlines()

assert len(all_mask) == len(tsv_lines)

f_w = open(os.path.join(database_tsv_dir, 'new_all.tsv'), 'w')
for mask, tsv in zip(all_mask, tsv_lines):
    if mask == 1:
        f_w.write(tsv)
f_w.close()

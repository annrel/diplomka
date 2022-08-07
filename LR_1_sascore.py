#!/usr/bin/env python3 

import molpher
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs, rdBase
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
import time
DrawingOptions.includeAtomNumbers=True

from molpher.core import ExplorationTree as ETree
from molpher.core import MolpherMol

loaded_model = pickle.load(open('finalized_model_LR.sav', 'rb'))

qsar_model = loaded_model

def predict_with_model(morph):
    morph.dist_to_target = qsar_model.predict_proba([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(morph.getSMILES()), 2, nBits=2048)])[0][0]
    return morph.dist_to_target


list_of_start_mols = ['c1ccc2c(c1)NNC2']

for molecule in list_of_start_mols:

    start_mol = MolpherMol(molecule) # some start molecule
    tree = ETree.create(source=start_mol)

    tree.params = {
        'accept_min' : 500,
        'accept_max' : 2000,
        'sascoreMax' : 4.0
    }
    tree.setThreadCount(2)
    print("Number of threads: ", tree.getThreadCount())

#qsar_model = YourModel() # we initalize it ahead of time
    dis_tar = []

    time_iter = []
    i = 0
    morph = []
    prediction = []

    while i < 100:
        start_time = time.perf_counter()
        print('Generating...')
        tree.generateMorphs() # vcetne predikce a zapsani do dist_to_target
        print(len(tree.candidates))
        print("Sorting...")
        tree.sortMorphs()
        print('Filtering...')
        tree.filterMorphs()
        print("Generating new mask...")
        mask = tree.candidates_mask
        mask = [predict_with_model(x) < 0.5 if mask[idx] else False for idx,x in enumerate(tree.candidates)]
        tree.candidates_mask = mask
        print("Saving data...")
        dis_tar.append((i, [x.dist_to_target for idx,x in enumerate(tree.candidates) if tree.candidates_mask[idx]]))
        print("Extending and pruning...")
        tree.extend()
        tree.prune()
        print("Getting leaves info...")
        leaves = [(x.smiles, x.dist_to_target, x.sascore) for x in tree.leaves]
        print(f'Number of leaves: {len(leaves)}')
        print(f'Average distance in leaves: {sum([x[1] for x in leaves]) / len(leaves)}')
        print("Collecting leves data...")
        morph.append([x[0] for x in leaves])
        prediction.append([x[1] for x in leaves])
        print("Time per iteration: ", time.perf_counter() - start_time)
        i += 1

    run={'morph':morph , 'prediction': prediction}
    run=pd.DataFrame(run)
    run.to_csv(f'From_{molecule}_with_model_LR_sascore.csv',sep='\t', index=False)

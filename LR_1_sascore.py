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



list_of_start_mols = ['c1ccc2c(c1)NNC2']

for molecule in list_of_start_mols:

    start_mol = MolpherMol(molecule) # some start molecule
    tree = ETree.create(source=start_mol)

    tree.params = {
        'accept_min' : 500,
        'accept_max' : 2000,
        'sascoreMax' : 4.0
    }

#qsar_model = YourModel() # we initalize it ahead of time
    dis_tar = []

    time_iter = []
    i = 0
    morph = []
    prediction = []

    while i < 100:
        start_time = time.time()
        tree.generateMorphs() # vcetne predikce a zapsani do dist_to_target
        print(len(tree.candidates))
        tree.sortMorphs()
        tree.filterMorphs()
        [predict_with_model(x) for x in tree.candidates]
        new_mask = [True if idx < 1000 else False for idx, x in enumerate(tree.candidates_mask)]
        tree.candidates_mask = new_mask
        dis_tar.append((i, [x.dist_to_target for idx,x in enumerate(tree.candidates) if tree.candidates_mask[idx]]))
        tree.extend()
        tree.prune()
        tree.leaves 
        i += 1
        morph.append([x.getSMILES() for x in tree.leaves])
        prediction.append([x.dist_to_target for x in tree.leaves])
        time_iter.append(time.time() - start_time)

    run={'morph':morph , 'prediction': prediction}
    run=pd.DataFrame(run)
    run.to_csv(f'From_{molecule}_with_model_LR_sascore.csv',sep='\t', index=False)
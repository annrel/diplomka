#!/usr/bin/env python3
import os.path
import pickle
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem
import time

from molpher.core import ExplorationTree as ETree
from molpher.core import MolpherMol

loaded_model = pickle.load(open('finalized_model_LR.sav', 'rb'))

qsar_model = loaded_model

def predict_with_model(morph):
    return qsar_model.predict_proba([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(morph.getSMILES()), 2, nBits=2048)])[0][0]


list_of_start_mols = ['c1ccc2c(c1)NNC2']

class StopWatch:

    def __init__(self):
        self.start = time.perf_counter()

    def reset(self):
        self.start = time.perf_counter()

    def stop(self, msg='Time it took: '):
        ret = time.perf_counter() - self.start
        print(msg + str(ret))
        self.reset()
        return ret

for molecule in list_of_start_mols:
    outfile = f'From_{molecule}_with_model_LR_sascore.csv'
    if os.path.exists(outfile):
        os.remove(outfile)

    start_mol = MolpherMol(molecule) # some start molecule
    tree = ETree.create(source=start_mol)

    tree.params = {
        'accept_min' : 500,
        'accept_max' : 2000,
        'sascoreMax' : 4.0
    }
    tree.setThreadCount(6)
    print("Number of threads: ", tree.getThreadCount())

    i = 0
    while i < 10:
        print(f"========= Iteration {i+1} =========")
        watch = StopWatch()
        print('Generating...')
        tree.generateMorphs()
        candidates = tree.candidates
        print(f"Generated {len(candidates)} molecules.")
        watch.stop()
        print("Sorting...")
        tree.sortMorphs()
        watch.stop()
        print('Filtering...')
        tree.filterMorphs()
        watch.stop()
        print("Predicting...")
        mask = list(tree.candidates_mask)
        predictions = [predict_with_model(x) if mask[idx] else False for idx,x in enumerate(candidates)]
        watch.stop()
        print("Masking...")
        for idx,pred in enumerate(predictions):
            if pred and pred < 0.5:
                tree.candidates[idx].dist_to_target = pred
                # print(tree.candidates[idx].dist_to_target)
                mask[idx] = True
            else:
                mask[idx] = False
        tree.candidates_mask = mask
        watch.stop()
        print("Extending and pruning...")
        tree.extend()
        tree.prune()
        watch.stop()
        print("Getting leaves info...")
        leaves = [(x.smiles, x.dist_to_target, x.sascore) for x in tree.leaves]
        # print(sum(mask), len(leaves))
        # print(leaves)
        print(f'Number of leaves: {len(leaves)}')
        print(f'Average distance in leaves: {sum([x[1] for x in leaves]) / len(leaves)}')
        print("Collecting leaves data...")
        leaves_smiles = [x[0] for x in leaves]
        leaves_dists = [x[1] for x in leaves]
        leaves_sascores = [x[2] for x in leaves]
        watch.stop()
        print("Saving iteration data...")
        df = pd.DataFrame({'leaf': leaves_smiles , 'distance': leaves_dists, 'sascore': leaves_sascores, 'iteration': [i+1] * len(leaves_smiles)})
        df.to_csv(outfile, index=False, header=True if i == 0 else False, mode='a')
        watch.stop()

        i += 1
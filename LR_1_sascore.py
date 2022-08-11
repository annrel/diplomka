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


molecule = 'c1ccc2c(c1)NNC2' # TODO: get it from command line when ran for real

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

prefix = f'{molecule}_LR_sascore'
outfile = f'{prefix}.csv'

start_mol = MolpherMol(molecule) # some start molecule
template_file = f'{prefix}_snapshot.xml'
if os.path.exists(template_file):
    tree = ETree.create(template_file)
else:
    tree = ETree.create(source=start_mol)

tree.params = {
    'accept_min' : 500,
    'accept_max' : 2000,
    'sascoreMax' : 4.0
}
tree.setThreadCount(12)
print("Number of threads: ", tree.getThreadCount())

i = 0
iterfile = f"{prefix}_iter.txt"
if os.path.exists(iterfile):
    with open(iterfile, mode="r", encoding="utf-8") as _file:
        i = int(_file.read())
first_run = i == 0
dist_threshold = 0.5
while i < 100:
    i += 1
    print(f"========= Iteration {i} =========")
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
    candidates = tree.candidates
    for idx,pred in enumerate(predictions):
        if pred and pred < dist_threshold:
            candidates[idx].dist_to_target = pred
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
    average = sum([x[1] for x in leaves]) / len(leaves)
    maximum = max([x[1] for x in leaves])
    print(f'Average distance in leaves: {average}')
    print(f'Maximum distance in leaves: {maximum}')
    # new_threshold = min(1.2 * maximum, dist_threshold)
    # if new_threshold < dist_threshold:
    #     dist_threshold = new_threshold
    #     print(f"Switching to new threshold: {dist_threshold}")
    print("Collecting leaves data...")
    leaves_smiles = [x[0] for x in leaves]
    leaves_dists = [x[1] for x in leaves]
    leaves_sascores = [x[2] for x in leaves]
    watch.stop()
    print("Saving iteration data...")
    df = pd.DataFrame({'leaf': leaves_smiles , 'distance': leaves_dists, 'sascore': leaves_sascores, 'iteration': [i] * len(leaves_smiles)})
    df.to_csv(outfile, index=False, header=True if first_run and i == 1 else False, mode='a')
    watch.stop()
    print("Saving tree...")
    tree.save(template_file)
    with open(iterfile, mode="w", encoding="utf-8") as _file:
        _file.write(str(i))
    watch.stop()
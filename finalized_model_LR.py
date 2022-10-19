#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
import pandas.io.sql as pdsql
import collections
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs, rdBase
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from sklearn.model_selection import StratifiedKFold
import random
import operator
import math
import copy
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pickle

s = 42

data=pd.read_csv('data_including_ch24.csv',sep='\t')
fps_data = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=2048) for x in data.smiles]
#smt = SMOTE(random_state = s)
y=np.asarray(data.active.tolist(),dtype="float")
X = np.array(fps_data,dtype="float") 
#X_smt, y_smt = smt.fit_resample(X, y)
model = LogisticRegression(random_state=42, solver='liblinear')

model.fit(X, y)

filename = 'finalized_model_LR.sav'
pickle.dump(model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))

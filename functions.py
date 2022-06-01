#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
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
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE 

class BaseModel:
    
    def __init__(self, data, fps_data, seeds, balancer=None):
        self.y = np.asarray(data.active.tolist(),dtype="float")
        self.X = np.array(fps_data,dtype="float")
        self.balancer = balancer 
        self.seeds = seeds
        
             
        

    #function that makes search through parameter's grid and returns the best parametrs of the machine learning model
    def grid_search(self, X_train, y_train, n_it, cv):
        est = RandomizedSearchCV(estimator = self.est, param_distributions = self.random_grid, n_iter = n_it, cv = cv, verbose=2, n_jobs = -1)
        est.fit(X_train, y_train)
        return est.best_estimator_
    
    #split data into train and test
    def data_split(self, seed, test_size=0.2):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed, stratify=self.y)

    #applies balancing and calls grid_search and evaluation functions
    def predict(self, n_it, splits):
        accur = []
        mcc = []
        for seed in self.seeds:
            X_train, X_test, y_train, y_test = self.data_split(seed)
            if self.balancer is not None:
                X_train, y_train = self.balancer.fit_resample(X_train, y_train)
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
            best_estimator = self.grid_search(X_train, y_train, n_it, cv)
            print('For seed: ', seed, 'best parameters: ', best_estimator.get_params)
            evaluation = self.evaluate(best_estimator, X_test, y_test)
            accur.append(evaluation['accuracy'])
            mcc.append(evaluation['mattews_coef'])
            print('Accuracy: ', evaluation['accuracy'], 'MCC: ', evaluation['mattews_coef'])

        print('Average accuracy: ', '{0:0.2f}'.format(mean(accur)), 'ACC std:', '{0:0.2f}'.format(np.std(accur)), 'Average MCC: ', '{0:0.2f}'.format(mean(mcc)), 'MCC std:', '{0:0.2f}'.format(np.std(mcc)) )

        with open('{}.txt'.format(case), 'w') as f:
            print('Average accuracy: ', '{0:0.2f}'.format(mean(accur)), 'ACC std:', '{0:0.2f}'.format(np.std(accur)), 'Average MCC: ', '{0:0.2f}'.format(mean(mcc)), 'MCC std:', '{0:0.2f}'.format(np.std(mcc)) )
    
    #evaluate best model performance
    def evaluate(self, best_estimator, X_test, y_test):
        predictions = best_estimator.predict(X_test)
        probabilities = best_estimator.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test, probabilities[:, 1])
        conf_mat = confusion_matrix(y_test, predictions)
        accuracy = metrics.accuracy_score(y_test, predictions)
        presision = metrics.precision_score(y_test, predictions)
        matthews_coef = matthews_corrcoef(y_test, predictions)
        recall = metrics.recall_score(y_test, predictions)
        return {'conf_mat': conf_mat, 
                'probabilities': probabilities, 
                'accuracy': accuracy, 
                'presision': presision, 
                'mattews_coef': mattews_coef, 
                'recall': recall, 
                'auc': auc
               }

class RFModel(BaseModel):
    def __init__(self, data, fps_data, seeds, balancer=None):
        super().__init__(data, fps_data, balancer=balancer)
        n_estimators = [50, 100, 300, 500]
        max_features = [102,256,512,768,1024]
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        criterion = ['gini', 'entropy']
        self.random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'random_state': seeds,
                   'criterion': criterion}

        self.est = RandomForestClassifier()


class KNNModel(BaseModel):
    def __init__(self, data, fps_data, balancer=None):
        super().__init__(data, fps_data, balancer=balancer)
        n_neighbors = [3,5,11,19]
        self.random_grid = {'n_neighbors': n_neighbors}
        
        self.est = KNeighborsClassifier()



class LRModel(BaseModel):
    def __init__(self, data, fps_data, seeds, balancer=None):
        super().__init__(data, fps_data, seeds, balancer=balancer)
        C = [100, 10, 1.0, 0.1, 0.01]
        penalty = ['l2']
        solver = ['newton-cg', 'lbfgs', 'liblinear']
        self.random_grid = {'C': C,
                   'penalty': penalty,
                   'random_state': seeds,
                   'solver': solver}

        
        self.est = LogisticRegression()

class SVMModel(BaseModel):
    def __init__(self, data, fps_data, seeds, balancer=None):
        super().__init__(data, fps_data, balancer=balancer)
        C = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        degree = [0, 1, 2, 3, 4, 5, 6]
        self.random_grid = {'C': C,
                   'degree': degree,
                   'random_state':seeds,
                   'kernel': kernel}


        self.est = SVC()
        
    def evaluate(self, best_estimator, X_test, y_test):
        predictions = best_estimator.predict(X_test)
        conf_mat = confusion_matrix(y_test, predictions)
        accuracy = metrics.accuracy_score(y_test, predictions)
        presision = metrics.precision_score(y_test, predictions)
        matthews_coef = matthews_corrcoef(y_test, predictions)
        recall = metrics.recall_score(y_test, predictions)
        return {'conf_mat': conf_mat, 
                'probabilities': probabilities, 
                'accuracy': accuracy, 
                'presision': presision, 
                'mattews_coef': mattews_coef, 
                'recall': recall, 
                'auc': auc
               }

class MLPModel(BaseModel):
    def __init__(self, data, fps_data, seeds, balancer=None):
        super().__init__(data, fps_data, balancer=balancer)
        hidden_layer_sizes = [(10,), (100,), (340,), (512,), (683,)]
        activation = ['logistic']
        solver = ['adam']
        alpha = [0.0001, 0.05]
        learning_rate = ['constant','adaptive']
        early_stopping =  [True]
        self.random_grid = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'random_state': seeds,
            'early_stopping': early_stopping
        }


        self.est = MLPClassifier()







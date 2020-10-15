from functools import partial
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import os
import sys

sys.path.append('../soln-ml/')
from solnml.datasets.utils import load_data

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)

args = parser.parse_args()
dataset_str = args.datasets


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, '../soln-ml/', False, task_type=0)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


grid = {'learning_rate': np.logspace(-2, np.log10(2), 20),
        'n_estimators': np.linspace(50, 487, 20).astype(int)}

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import balanced_accuracy_scorer

clf = GridSearchCV(AdaBoostClassifier(), grid, scoring=balanced_accuracy_scorer, n_jobs=2, cv=3)

dataset_list = dataset_str.split(',')
check_datasets(dataset_list)

for dataset in dataset_list:
    node = load_data(dataset, '../soln-ml/', True, task_type=0)
    _x, _y = node.data[0], node.data[1]
    clf.fit(_x, _y)
    results = clf.cv_results_
    params = results['params']
    scores = results['mean_test_score']
    # print(params,scores)
    length = len(params)
    d = [(params[i], scores[i]) for i in range(length)]

    with open('%s-adaboost.pkl' % dataset, 'wb')as f:
        pickle.dump(d, f)

from functools import partial
import numpy as np
import sys
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append('../soln-ml')
from solnml.datasets.utils import load_data

import pickle
import argparse
from litebo.facade.bo_facade import BayesianOptimization as BO

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--mode', type=str, default='bo')

args = parser.parse_args()
dataset_str = args.datasets
run_count = args.n
mode = args.mode


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, '../soln-ml/', False, task_type=4)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


def check_true(p):
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p):
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p):
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p):
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))


def get_cs():
    cs = ConfigurationSpace()
    n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 1023, default_value=31)
    learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
    min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
    subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
    reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
    reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                            colsample_bytree, reg_alpha, reg_lambda])
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = LightGBM(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred)


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, random_state=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree

        self.n_jobs = 1
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.estimator = LGBMRegressor(num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       min_child_weight=self.min_child_weight,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha,
                                       reg_lambda=self.reg_lambda,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)


dataset_list = dataset_str.split(',')
check_datasets(dataset_list)
cs = get_cs()

_run_count = min(int(len(set(cs.sample_configuration(30000))) * 0.75), run_count)
print(_run_count)

for dataset in dataset_list:
    node = load_data(dataset, '../soln-ml/', True, task_type=4)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()
    with open('logs/rgs_%s-lightgbm-%s-%d.pkl' % (dataset, mode, run_count), 'wb')as f:
        pickle.dump(bo.get_history().data, f)

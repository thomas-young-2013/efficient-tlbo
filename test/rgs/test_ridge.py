from functools import partial
import numpy as np
import sys
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
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
    alpha = UniformFloatHyperparameter("alpha", 0.01, 32, log=True, default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                     log=True)

    max_iter = UniformFloatHyperparameter("max_iter", 100, 1000, q=100, default_value=100)
    solver = CategoricalHyperparameter("solver", choices=["auto", "saga"], default_value="auto")

    cs = ConfigurationSpace()
    cs.add_hyperparameters([alpha, tol, max_iter, solver])
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = RidgeRegressor(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred)


class RidgeRegressor:
    def __init__(self, alpha, solver, tol, max_iter, random_state=None):
        self.alpha = alpha
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        from sklearn.linear_model import Ridge
        self.estimator = Ridge(alpha=self.alpha,
                               tol=self.tol,
                               max_iter=self.max_iter,
                               solver=self.solver,
                               random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
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
    with open('logs/rgs_%s-ridge_regression-%s-%d.pkl' % (dataset, mode, run_count), 'wb')as f:
        pickle.dump(bo.get_history().data, f)

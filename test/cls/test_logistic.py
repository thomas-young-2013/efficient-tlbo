from functools import partial
import numpy as np
import sys
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

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
            _ = load_data(_dataset, '../soln-ml/', False, task_type=0)
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
    C = UniformFloatHyperparameter("C", 0.03125, 10, log=True,
                                   default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                     log=True)

    max_iter = UnParametrizedHyperparameter("max_iter", 3000)

    penalty = CategoricalHyperparameter(name="penalty",
                                        choices=["l1", "l2"],
                                        default_value="l2")
    solver = CategoricalHyperparameter(name="solver", choices=["liblinear", "saga"], default_value="liblinear")

    cs = ConfigurationSpace()
    cs.add_hyperparameters([C, penalty, solver, tol, max_iter])
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = Logistic_Regression(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)


class Logistic_Regression:
    def __init__(self, C, penalty, solver, tol, max_iter, random_state=None):
        self.C = C
        self.tol = tol
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.estimator = None
        self.time_limit = None

        self.n_jobs = 1

    def fit(self, X, Y):
        from sklearn.linear_model import LogisticRegression

        self.C = float(self.C)

        self.estimator = LogisticRegression(random_state=self.random_state,
                                            solver=self.solver,
                                            penalty=self.penalty,
                                            multi_class='ovr',
                                            C=self.C,
                                            tol=self.tol,
                                            max_iter=self.max_iter,
                                            n_jobs=self.n_jobs)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)


dataset_list = dataset_str.split(',')
check_datasets(dataset_list)
cs = get_cs()

_run_count = min(int(len(set(cs.sample_configuration(30000))) * 0.75), run_count)
print(_run_count)

for dataset in dataset_list:
    node = load_data(dataset, '../soln-ml/', True, task_type=0)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()
    with open('logs/%s-logistic_regression-%s-%d.pkl' % (dataset, mode, run_count), 'wb')as f:
        pickle.dump(bo.get_history().data, f)

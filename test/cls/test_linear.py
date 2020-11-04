from functools import partial
import numpy as np
import sys
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
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
    cs = ConfigurationSpace()
    penalty = CategoricalHyperparameter(
        "penalty", ["l1", "l2"], default_value="l2")
    loss = CategoricalHyperparameter(
        "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
    dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
    # This is set ad-hoc
    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    C = UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0)
    multi_class = Constant("multi_class", "ovr")
    # These are set ad-hoc
    fit_intercept = Constant("fit_intercept", "True")
    intercept_scaling = Constant("intercept_scaling", 1)
    cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                            fit_intercept, intercept_scaling])

    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(penalty, "l1"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    constant_penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, "False"),
        ForbiddenEqualsClause(penalty, "l2"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    penalty_and_dual = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, "True"),
        ForbiddenEqualsClause(penalty, "l1")
    )
    cs.add_forbidden_clause(penalty_and_loss)
    cs.add_forbidden_clause(constant_penalty_and_loss)
    cs.add_forbidden_clause(penalty_and_dual)
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = LibLinear_SVC(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)


class LibLinear_SVC:
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.svm
        import sklearn.multiclass

        # In case of nested penalty
        if isinstance(self.penalty, dict):
            combination = self.penalty
            self.penalty = combination['penalty']
            self.loss = combination['loss']
            self.dual = combination['dual']

        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        if check_none(self.class_weight):
            self.class_weight = None

        estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                          loss=self.loss,
                                          dual=self.dual,
                                          tol=self.tol,
                                          C=self.C,
                                          class_weight=self.class_weight,
                                          fit_intercept=self.fit_intercept,
                                          intercept_scaling=self.intercept_scaling,
                                          multi_class=self.multi_class,
                                          random_state=self.random_state)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
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
    node = load_data(dataset, '../soln-ml/', True, task_type=0)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()
    with open('logs/%s-liblinear_svc-%s-%d.pkl' % (dataset, mode, run_count), 'wb')as f:
        pickle.dump(bo.get_history().data, f)

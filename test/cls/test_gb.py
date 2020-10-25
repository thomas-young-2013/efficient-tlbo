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
    cs = ConfigurationSpace()
    loss = Constant("loss", "deviance")
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
    # n_estimators = UniformIntegerHyperparameter(
    #     "n_estimators", 50, 500, default_value=100)
    n_estimators = Constant("n_estimators", 100)
    max_depth = UniformIntegerHyperparameter(
        name="max_depth", lower=1, upper=8, default_value=3)
    criterion = CategoricalHyperparameter(
        'criterion', ['friedman_mse', 'mse'],
        default_value='mse')
    min_samples_split = UniformIntegerHyperparameter(
        name="min_samples_split", lower=2, upper=20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        name="min_samples_leaf", lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
    subsample = UniformFloatHyperparameter(
        name="subsample", lower=0.01, upper=1.0, default_value=1.0)
    max_features = UniformFloatHyperparameter(
        "max_features", 0.1, 1.0, default_value=1)
    max_leaf_nodes = UnParametrizedHyperparameter(
        name="max_leaf_nodes", value="None")
    min_impurity_decrease = UnParametrizedHyperparameter(
        name='min_impurity_decrease', value=0.0)
    cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                            criterion, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, subsample,
                            max_features, max_leaf_nodes,
                            min_impurity_decrease])
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = GradientBoostingClassifier(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)


class GradientBoostingClassifier:
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, criterion, max_features,
                 max_leaf_nodes, min_impurity_decrease, random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import GradientBoostingClassifier

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            self.max_features = float(self.max_features)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.min_impurity_decrease = float(self.min_impurity_decrease)
            self.verbose = int(self.verbose)

            self.estimator = GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )

        self.estimator.fit(X, y, sample_weight=sample_weight)

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

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
    node = load_data(dataset, '../soln-ml/', True, task_type=0)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()
    with open('logs/%s-gradient_boosting-%s-%d.pkl' % (dataset, mode, run_count), 'wb')as f:
        pickle.dump(bo.get_history().data, f)

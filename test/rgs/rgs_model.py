import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from ConfigSpace.conditions import EqualsCondition, InCondition


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


class AdaboostRegressor:

    def __init__(self, n_estimators, learning_rate, max_depth,
                 random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y, sample_weight=None):
        from sklearn.ensemble import AdaBoostRegressor as ABR
        from sklearn.tree import DecisionTreeRegressor
        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = DecisionTreeRegressor(max_depth=self.max_depth)

        estimator = ABR(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )

        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False)

        cs.add_hyperparameters([n_estimators, learning_rate, max_depth])
        return cs


class ExtraTreesRegressor:

    def __init__(self, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, **kwargs):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion

        if check_none(max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
        if check_none(max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.bootstrap = check_for_bool(bootstrap)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_estimators

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesRegressor as ETR

        max_features = int(X.shape[1] ** float(self.max_features))
        self.estimator = ETR(n_estimators=self.get_max_iter(),
                             criterion=self.criterion,
                             max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             min_samples_leaf=self.min_samples_leaf,
                             bootstrap=self.bootstrap,
                             max_features=max_features,
                             max_leaf_nodes=self.max_leaf_nodes,
                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                             min_impurity_decrease=self.min_impurity_decrease,
                             oob_score=self.oob_score,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             random_state=self.random_state,
                             warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "mae"], default_value="mse")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1., default_value=0.5)

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False")
        cs.add_hyperparameters([criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease, bootstrap])

        return cs


class GradientBoostingRegressor:
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, criterion, max_features,
                 max_leaf_nodes, min_impurity_decrease, random_state=None,
                 verbose=0, **kwargs):
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
        self.time_limit = None

    def fit(self, X, y, sample_weight=None):

        from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR
        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)

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

        self.estimator = GBR(
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

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        loss = CategoricalHyperparameter("loss", ['ls', 'lad'], default_value='ls')
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", 50, 500, default_value=200)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3)
        criterion = CategoricalHyperparameter(
            'criterion', ['friedman_mse', 'mse', 'mae'],
            default_value='friedman_mse')
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.1, upper=1.0, default_value=1.0)
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


class KNearestNeighborsRegressor:

    def __init__(self, n_neighbors, weights, p, random_state=None, n_jobs=1, **kwargs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.time_limit = None

    def fit(self, X, Y):
        from sklearn.neighbors import KNeighborsRegressor
        self.estimator = KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                             weights=self.weights,
                                             p=self.p,
                                             n_jobs=self.n_jobs)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])

        return cs


class LassoRegressor:
    def __init__(self, alpha, tol, max_iter, random_state=None, **kwargs):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        from sklearn.linear_model import Lasso
        self.estimator = Lasso(alpha=self.alpha,
                               tol=self.tol,
                               max_iter=self.max_iter,
                               random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        alpha = UniformFloatHyperparameter("alpha", 0.01, 32, log=True, default_value=1.0)
        tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                         log=True)

        max_iter = UniformFloatHyperparameter("max_iter", 100, 1000, q=100, default_value=100)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha, tol, max_iter])
        return cs


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, random_state=1, n_jobs=1, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree

        self.n_jobs = n_jobs
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

    @staticmethod
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


class LibSVM_SVR:
    def __init__(self, epsilon, C, kernel, gamma, shrinking, tol, max_iter,
                 degree=3, coef0=0, random_state=None, **kwargs):
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.svm import SVR

        # Nested kernel
        if isinstance(self.kernel, tuple):
            nested_kernel = self.kernel
            self.kernel = nested_kernel[0]
            if self.kernel == 'poly':
                self.degree = nested_kernel[1]['degree']
                self.coef0 = nested_kernel[1]['coef0']
            elif self.kernel == 'sigmoid':
                self.coef0 = nested_kernel[1]['coef0']

        self.epsilon = float(self.epsilon)
        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        self.estimator = SVR(epsilon=self.epsilon,
                             C=self.C,
                             kernel=self.kernel,
                             degree=self.degree,
                             gamma=self.gamma,
                             coef0=self.coef0,
                             shrinking=self.shrinking,
                             tol=self.tol,
                             max_iter=self.max_iter)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        epsilon = CategoricalHyperparameter("epsilon", [1e-4, 1e-3, 1e-2, 1e-1, 1], default_value=1e-4)
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                       default_value=1.0)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel",
                                           choices=["rbf", "poly", "sigmoid"],
                                           default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                              default_value="True")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                         log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", 2000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([epsilon, C, kernel, degree, gamma, coef0,
                                shrinking, tol, max_iter])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs


class LibLinear_SVR:
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, epsilon, loss, dual, tol, C,
                 fit_intercept, intercept_scaling,
                 random_state=None, **kwargs):
        self.epsilon = epsilon
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.svm import LinearSVR

        # In case of nested loss
        if isinstance(self.loss, dict):
            combination = self.loss
            self.loss = combination['loss']
            self.dual = combination['dual']

        self.epsilon = float(self.epsilon)
        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        self.estimator = LinearSVR(epsilon=self.epsilon,
                                   loss=self.loss,
                                   dual=self.dual,
                                   tol=self.tol,
                                   C=self.C,
                                   fit_intercept=self.fit_intercept,
                                   intercept_scaling=self.intercept_scaling,
                                   random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        epsilon = CategoricalHyperparameter("epsilon", [1e-4, 1e-3, 1e-2, 1e-1, 1], default_value=1e-4)
        loss = CategoricalHyperparameter(
            "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"], default_value="epsilon_insensitive")
        dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        C = UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default_value=1.0)
        fit_intercept = Constant("fit_intercept", "True")
        intercept_scaling = Constant("intercept_scaling", 1)
        cs.add_hyperparameters([epsilon, loss, dual, tol, C,
                                fit_intercept, intercept_scaling])

        dual_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(loss, "epsilon_insensitive")
        )
        cs.add_forbidden_clause(dual_and_loss)
        return cs


class RandomForest:
    def __init__(self, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1, **kwargs):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = None
        self.time_limit = None

    @staticmethod
    def get_max_iter():
        return 100

    def get_current_iter(self):
        return self.estimator.n_estimators

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestRegressor

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

            if self.max_features not in ("sqrt", "log2", "auto"):
                max_features = int(X.shape[1] ** float(self.max_features))
            else:
                max_features = self.max_features

            self.bootstrap = check_for_bool(self.bootstrap)

            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_impurity_decrease = float(self.min_impurity_decrease)

            # initial fit of only increment trees
            self.estimator = RandomForestRegressor(
                n_estimators=self.get_max_iter(),
                criterion=self.criterion,
                max_features=max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "mae"], default_value="mse")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1., default_value=0.5)

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")
        cs.add_hyperparameters([criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                bootstrap, min_impurity_decrease])
        return cs


class RidgeRegressor:
    def __init__(self, alpha, solver, tol, max_iter, random_state=None, **kwargs):
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

    @staticmethod
    def get_cs():
        alpha = UniformFloatHyperparameter("alpha", 0.01, 32, log=True, default_value=1.0)
        tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                         log=True)

        max_iter = UniformFloatHyperparameter("max_iter", 100, 1000, q=100, default_value=100)
        solver = CategoricalHyperparameter("solver", choices=["auto", "saga"], default_value="auto")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha, tol, max_iter, solver])
        return cs

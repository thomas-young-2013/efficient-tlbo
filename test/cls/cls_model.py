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


class AdaboostClassifier:

    def __init__(self, n_estimators, learning_rate, algorithm, max_depth,
                 random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.tree
        from sklearn.ensemble import AdaBoostClassifier

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)

        estimator = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )

        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=8, default_value=3, log=False)

        cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        return cs


class ExtraTreesClassifier:

    def __init__(self, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None, **kwargs):

        self.n_estimators = self.get_max_iter()
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
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
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_estimators

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        if self.estimator is None:
            max_features = int(X.shape[1] ** float(self.max_features))
            self.estimator = ETC(n_estimators=self.get_max_iter(),
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
                                 class_weight=self.class_weight,
                                 warm_start=True)

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

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")

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


class GradientBoostingClassifier:
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

    @staticmethod
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


class KNearestNeighborsClassifier:

    def __init__(self, n_neighbors, weights, p, random_state=None, n_jobs=1, **kwargs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.multiclass

        estimator = \
            sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights=self.weights,
                                                   p=self.p,
                                                   n_jobs=self.n_jobs)

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

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

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


class LDA:
    def __init__(self, shrinkage, n_components, tol, shrinkage_factor=0.5, random_state=None, **kwargs):
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.tol = tol
        self.shrinkage_factor = shrinkage_factor
        self.estimator = None
        self.time_limit = None
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.multiclass
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # In case of nested shrinkage
        if isinstance(self.shrinkage, tuple):
            self.shrinkage_factor = self.shrinkage[1]['shrinkage_factor']
            self.shrinkage = self.shrinkage[0]

        if check_none(self.shrinkage):
            self.shrinkage_ = None
            solver = 'svd'
        elif self.shrinkage == "auto":
            self.shrinkage_ = 'auto'
            solver = 'lsqr'
        elif self.shrinkage == "manual":
            self.shrinkage_ = float(self.shrinkage_factor)
            solver = 'lsqr'
        else:
            raise ValueError(self.shrinkage)

        self.n_components = int(self.n_components)
        self.tol = float(self.tol)

        estimator = LinearDiscriminantAnalysis(
            n_components=self.n_components, shrinkage=self.shrinkage_,
            tol=self.tol, solver=solver)

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

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        shrinkage = CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default_value="None")
        shrinkage_factor = UniformFloatHyperparameter(
            "shrinkage_factor", 0., 1., 0.5)
        n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        cs.add_hyperparameters([shrinkage, shrinkage_factor, n_components, tol])

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, random_state=None, n_jobs=1, **kwargs):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
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
        n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
        max_depth = Constant('max_depth', 15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
        min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
        subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
        cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                                colsample_bytree])
        return cs


class LibSVM_SVC:
    def __init__(self, C, kernel, gamma, shrinking, tol, max_iter,
                 class_weight=None, degree=3, coef0=0, random_state=None, **kwargs):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.svm
        # Nested kernel
        if isinstance(self.kernel, tuple):
            nested_kernel = self.kernel
            self.kernel = nested_kernel[0]
            if self.kernel == 'poly':
                self.degree = nested_kernel[1]['degree']
                self.coef0 = nested_kernel[1]['coef0']
            elif self.kernel == 'sigmoid':
                self.coef0 = nested_kernel[1]['coef0']

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

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         shrinking=self.shrinking,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         max_iter=self.max_iter,
                                         random_state=self.random_state,
                                         decision_function_shape='ovr')
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
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
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                              default_value="True")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                         log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", 10000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                                tol, max_iter])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs


class LibLinear_SVC:
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None, **kwargs):
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

    @staticmethod
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


class Logistic_Regression:
    def __init__(self, C, penalty, solver, tol, max_iter, random_state=None, n_jobs=1, **kwargs):
        self.C = C
        self.tol = tol
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.estimator = None
        self.time_limit = None

        self.n_jobs = n_jobs

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

    @staticmethod
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


class QDA:

    def __init__(self, reg_param, random_state=None, **kwargs):
        if reg_param is not None:
            self.reg_param = float(reg_param)
        else:
            self.reg_param = None
        self.estimator = None
        self.time_limit = None
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.discriminant_analysis

        estimator = sklearn.discriminant_analysis. \
            QuadraticDiscriminantAnalysis(reg_param=self.reg_param)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            problems = []
            for est in self.estimator.estimators_:
                problem = np.any(np.any([np.any(s <= 0.0) for s in
                                         est.scalings_]))
                problems.append(problem)
            problem = np.any(problems)
        else:
            problem = np.any(np.any([np.any(s <= 0.0) for s in
                                     self.estimator.scalings_]))
        if problem:
            raise ValueError('Numerical problems in QDA. QDA.scalings_ '
                             'contains values <= 0.0')
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0,
                                               default_value=0.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(reg_param)
        return cs


class RandomForest:
    def __init__(self, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1,
                 class_weight=None, **kwargs):
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
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 100

    def get_current_iter(self):
        return self.estimator.n_estimators

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier

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
            self.estimator = RandomForestClassifier(
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
                class_weight=self.class_weight,
                warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")

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

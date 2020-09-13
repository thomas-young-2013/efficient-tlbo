import abc
import time
import typing
import numpy as np
from typing import List

from tlbo.model.util_funcs import get_types
from tlbo.model.model_builder import build_model
from tlbo.utils.constants import VERY_SMALL_NUMBER
from tlbo.config_space import ConfigurationSpace, Configuration
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class BaseFacade(object):
    def __init__(self, config_space: ConfigurationSpace,
                 source_hpo_data: List,
                 rng: np.random.RandomState,
                 target_hp_configs: List = None,
                 history_dataset_features: List = None,
                 num_src_hpo_trial: int=50,
                 surrogate_type='gp_mcmc'):
        self.config_space = config_space
        self.rng = rng
        # The number of source problems.
        self.K = len(source_hpo_data)
        self.num_src_hpo_trial = num_src_hpo_trial
        self.source_hpo_data = source_hpo_data
        self.target_hp_configs = target_hp_configs
        self.source_surrogates = None
        self.target_surrogate = None
        self.history_dataset_features = history_dataset_features
        if history_dataset_features is not None:
            assert len(history_dataset_features) == self.K
        self.surrogate_type = surrogate_type

        self.types, self.bounds = get_types(config_space)
        self.instance_features = None
        self.var_threshold = VERY_SMALL_NUMBER
        self.w = None

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def build_source_surrogates(self):
        print('start to train base surrogates.')
        start_time = time.time()
        self.source_surrogates = list()
        for hpo_evaluation_data in self.source_hpo_data:
            print('.', end='')
            model = build_model(self.surrogate_type, self.config_space, self.rng)
            _X, _y = list(), list()
            for _config, _config_perf in hpo_evaluation_data.items():
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)
            X = X[:self.num_src_hpo_trial]
            y = y[:self.num_src_hpo_trial]

            # Prevent the same value in y.
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
            model.train(X, y)
            self.source_surrogates.append(model)
        print()
        print('Building base surrogates took %.3fs.' % (time.time() - start_time))

    def build_single_surrogate(self, X: np.ndarray, y: np.array):
        model = build_model(self.surrogate_type, self.config_space, self.rng)
        # Prevent the same value in y.
        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_mean_unit_var_normalization(y)
        model.train(X, y)
        return model

    def predict_marginalized_over_instances(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.bounds), X.shape[1]))

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self.predict(X)
            assert var is not None  # please mypy

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            n_instances = len(self.instance_features)

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_ = np.hstack(
                (np.tile(x, (n_instances, 1)), self.instance_features))
            means, vars = self.predict(X_)
            assert vars is not None  # please mypy
            # VAR[1/n (X_1 + ... + X_n)] =
            # 1/n^2 * ( VAR(X_1) + ... + VAR(X_n))
            # for independent X_1 ... X_n
            var_x = np.sum(vars) / (len(vars) ** 2)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = np.mean(means)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var

    def combine_predictions(self, X: np.array, combination_method: str = 'idp_lc'):
        n, m = X.shape[0], len(self.w)
        mu, var = np.zeros((n, 1)), np.zeros((n, 1))
        w = self.w
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))

        target_var = None
        # Predictions from source surrogates.
        for i in range(0, self.K + 1):
            if i == self.K:
                if self.target_surrogate is None:
                    _mu, _var = np.zeros((n, 1)), np.zeros((n, 1))
                else:
                    _mu, _var = self.target_surrogate.predict(X)
                target_var = _var
            else:
                _mu, _var = self.source_surrogates[i].predict(X)
            mu += w[i] * _mu
            var += w[i] * w[i] * _var

            # compute the gaussian experts.
            if combination_method == 'gpoe':
                var_buf[:, i] = (1./_var*w[i]).flatten()
                mu_buf[:, i] = (1./_var*_mu*w[i]).flatten()

        if combination_method == 'no_var':
            return mu, target_var
        elif combination_method == 'idp_lc':
            return mu, var
        elif combination_method == 'gpoe':
            tmp = np.sum(var_buf, axis=1)
            tmp[tmp == 0.] = 1e-5
            var = 1. / tmp
            mu = np.sum(mu_buf, axis=1) * var
            return mu, var
        else:
            raise ValueError('Invalid combination method %s.' % combination_method)

import numpy as np
import math
from tlbo.facade.base_surrogate import BaseSurrogate


class GPStackingSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern', alpha=1.):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.alpha = alpha
        self.configs_set = None
        self.cached_prior_mu = None
        self.cached_prior_sigma = None
        self.cached_stacking_mu = None
        self.cached_stacking_sigma = None
        self.prior_size = 0
        self.get_regressor(X_test=test_metadata[:, 1:])

    def get_regressor(self, X_test=None):
        # Collect the configs set.
        configs_list = list()
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            for item in X:
                item = list(item)
                if item not in configs_list:
                    configs_list.append(item)
        if X_test is not None:
            for item in X_test:
                item = list(item)
                if item not in configs_list:
                    configs_list.append(item)

        # Initialize mu and sigma vector.
        num_configs = len(configs_list)
        self.configs_set = configs_list
        self.cached_prior_mu = [0.] * num_configs
        self.cached_prior_sigma = [1.] * num_configs
        self.cached_stacking_mu = [0.] * num_configs
        self.cached_stacking_sigma = [1.] * num_configs

        # Train transfer learning regressor.
        self.prior_size = 0
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            self.train_regressor(X, y)
            self.prior_size = len(y)

    def train_regressor(self, X, y, is_top=False):
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)
        model = self.create_single_gp(lower, upper)

        # Get prior mu and sigma for configs in X.
        prior_mu = list()
        prior_sigma = list()
        for item in X:
            index = self.configs_set.index(list(item))
            prior_mu.append(self.cached_prior_mu[index])
            prior_sigma.append(self.cached_prior_sigma[index])
        prior_mu = np.array(prior_mu)
        prior_sigma = np.array(prior_sigma)
        # Training residual GP.
        model.train(X, y - prior_mu)

        # Update the prior surrogate: mu and sigma.
        top_size = len(y)
        for i, config in enumerate(self.configs_set):
            beta = self.alpha * top_size / (self.alpha * top_size + self.prior_size)
            mu_top, sigma_top = model.predict(np.array([config]))
            if is_top:
                self.cached_stacking_mu[i] = self.cached_prior_mu[i] + mu_top[0]
                self.cached_stacking_sigma[i] = math.pow(sigma_top[0], beta) * \
                                                math.pow(self.cached_prior_sigma[i], 1 - beta)
            else:
                self.cached_prior_mu[i] += mu_top[0]
                self.cached_prior_sigma[i] = math.pow(sigma_top[0], beta) * \
                                             math.pow(self.cached_prior_sigma[i], 1 - beta)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Decide whether to rebuild the transfer learning regressor.
        retrain = False
        for item in X:
            item = list(item)
            if item not in self.configs_set:
                retrain = True
                break
        if retrain:
            self.get_regressor(X_test=X)

        # Train the final regressor.
        self.train_regressor(X, y, is_top=True)

    def predict(self, X: np.array):
        mu_list, var_list = list(), list()
        for x in X:
            assert list(x) in self.configs_set
            index = self.configs_set.index(list(x))
            mu_list.append(self.cached_stacking_mu[index])
            var_list.append(self.cached_stacking_sigma[index])
        return np.array(mu_list), np.array(var_list)

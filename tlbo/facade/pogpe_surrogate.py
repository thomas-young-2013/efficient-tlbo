import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class POGPESurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)

    def train_experts(self, X_new, y_new):
        self.historical_model = list()
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
            X = np.r_[X, X_new]
            y = np.r_[y, y_new]
            # Scale the instance in training meta-dataset to [0, 1].
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            model = self.create_single_gp(lower, upper)
            model.train(X, y)
            self.historical_model.append(model)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)
        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_mean_unit_var_normalization(y)
        # Train the experts.
        self.train_experts(X, y)

    def predict(self, X: np.array):
        n = X.shape[0]
        m = self.historical_task_num
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))
        # Set beta = 1/M
        beta = 1./m
        # Predictions from basic surrogates.
        for i in range(0, self.historical_task_num):
            mu_t, var_t = self.historical_model[i].predict(X)
            var_buf[:, i] = 1. / var_t * beta
            mu_buf[:, i] = 1. / var_t * mu_t * beta

        tmp = np.sum(var_buf, axis=1)
        tmp[tmp == 0.] = 1e-5
        var = 1. / tmp
        mu = np.sum(mu_buf, axis=1) * var
        return mu, var

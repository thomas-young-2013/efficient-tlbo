import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.utils.scipy_solver_m2 import scipy_solve
from tlbo.utils.normalization import zero_one_normalization


class TransferSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern',
                 fusion=True, loss_type=0, v_type=0, lbd=0., alpha=1):
        BaseSurrogate.__init__(self, train_metadata, test_metadata, cov_amp=cov_amp, kernel_type=kernel_type,
                               normalize_output=False)
        # Initialize weights for all source surrogates.
        self.weights = np.array([1./self.historical_task_num] * self.historical_task_num)
        self.variance_type = v_type
        self.loss_type = loss_type
        self.use_fusion = fusion
        self.alpha = alpha
        self.lbd = lbd
        self.scale = True
        self.prior_size = self.train_metadata[0].shape[0]
        # Train each source surrogate.
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_one_normalization(y)
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            model = self.create_single_gp(lower, upper)
            model.train(X, y)
            self.historical_model.append(model)

    def batch_predict(self, X: np.ndarray):
        pred_y = None
        for i in range(0, self.historical_task_num):
            mu, _ = self.historical_model[i].predict(X)
            if pred_y is not None:
                pred_y = np.r_[pred_y, mu.reshape((1, -1))]
            else:
                pred_y = mu.reshape((1, -1))
        return pred_y

    def prior_surrogate_predict(self, X: np.ndarray):
        n = X.shape[0]
        m = self.weights.shape[0]
        mu, var = np.zeros(n), np.zeros(n)
        weights = self.weights
        max_index = np.argmax(weights)
        var_0 = var
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))
        # Predictions from basic surrogates.
        for i in range(0, self.historical_task_num):
            mu_t, var_t = self.historical_model[i].predict(X)
            mu += weights[i] * mu_t
            var += weights[i] * weights[i] * var_t
            if i == max_index:
                var_0 = var_t
            # compute the gaussian experts.
            var_buf[:, i] = 1./var_t*weights[i]
            mu_buf[:, i] = 1./var_t*mu_t*weights[i]

        if self.variance_type == 0:
            return mu, var_0
        elif self.variance_type == 1:
            return mu, var
        else:
            var = 1./np.sum(var_buf, axis=1)
            mu = np.sum(mu_buf, axis=1)*var
            return mu, var

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_one_normalization(y)
        self.update_scaled_incumbent(y)
        # First step: train a knowledge-based prior surrogate.
        # Predict the results on labeled data.
        pred_y = self.batch_predict(X)
        # Learn the weights.
        self.optimize(np.mat(pred_y).T, np.mat(y).T)

        if self.use_fusion:
            prior_y_mu, prior_y_var = self.prior_surrogate_predict(X)

            # Second step: train fusion surrogate between prior surrogate and the residual surrogate.
            # Train the current task's residual surrogate.
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            self.current_model = self.create_single_gp(lower, upper)
            self.current_model.train(X, y-prior_y_mu)

    def predict(self, X: np.array):
        prior_y_mu, prior_y_var = self.prior_surrogate_predict(X)
        if not self.use_fusion:
            return prior_y_mu, prior_y_var

        # Residual prediction from current surrogate.
        res_mu, res_var = self.current_model.predict(X)
        size = X.shape[0]
        beta = self.alpha * size / (self.alpha * size + self.prior_size)
        mu = prior_y_mu + res_mu
        var = np.power(res_var, beta) * np.power(prior_y_var, 1-beta)
        # var = res_var
        return mu, var

    def optimize(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, self.lbd, self.loss_type, debug=self._debug_mode)
        if status:
            x[x < 1e-3] = 0.
            self.weights = x

    def get_weights(self):
        return self.weights

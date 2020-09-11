import numpy as np
from rtl.facade.base_surrogate import BaseSurrogate
from rtl.utils.scipy_solver_m1 import scipy_solve, scipy_solve_rank
from rtl.utils.normalization import zero_one_normalization


class RobustTLSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern', v_type=3):
        BaseSurrogate.__init__(self, train_metadata, test_metadata, cov_amp=cov_amp, kernel_type=kernel_type,
                               normalize_output=False)
        # Initialize weights for all source surrogates.
        self.weights = np.array([1. / self.historical_task_num] * self.historical_task_num)
        self._norm = 2
        self._lbd = 0.
        self._mu = 1.
        self._rank_type = 3
        self.variance_type = v_type

        B, p = None, None
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
            if B is not None:
                B, p = np.r_[B, X], np.r_[p, y]
            else:
                B, p = X, y

        self.Bs = B
        self.ys = p
        self.B = None
        self.pred_fs = self.batch_predict(self.Bs)

    def batch_predict(self, X: np.ndarray):
        pred_y = None
        for i in range(0, self.historical_task_num):
            mu, _ = self.historical_model[i].predict(X)
            if pred_y is not None:
                pred_y = np.r_[pred_y, mu.reshape((1, -1))]
            else:
                pred_y = mu.reshape((1, -1))
        return pred_y

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Train the current task's surrogate and update the weights.
        # lower = np.amin(X, axis=0)
        # upper = np.amax(X, axis=0)
        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_one_normalization(y)
        # self.current_model = self.create_single_gp(lower, upper)
        # self.current_model.train(X, y)

        # Predict the results on labeled data.
        # mu, _ = self.current_model.predict(X)
        # pred_y = np.r_[pred_y, mu.reshape((1, -1))]
        pred_y = np.mat(self.batch_predict(X)).T
        assert pred_y.shape == (X.shape[0], self.weights.shape[0])

        # Predict the results on instances from source problems.
        # mu, _ = self.current_model.predict(self.Bs)
        # pred_y_source = np.mat(np.r_[self.pred_fs, mu.reshape((1, -1))]).T
        self.optimize(pred_y, np.mat(y).T, (np.mat(self.pred_fs).T, np.mat(self.ys).T))

    def predict(self, X: np.array):
        n = X.shape[0]
        m = self.weights.shape[0]
        mu, var = np.zeros(n), np.zeros(n)
        weights = self.weights
        max_index = np.argmax(self.weights)
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))

        # Prediction from current surrogate.
        var_1 = var

        # Predictions from basic surrogates.
        for i in range(0, self.historical_task_num):
            mu_t, var_t = self.historical_model[i].predict(X)
            mu += weights[i] * mu_t
            var += weights[i] * weights[i] * var_t
            if i == max_index:
                var_1 = var_t
            # compute the gaussian experts.
            var_buf[:, i] = 1./var_t*weights[i]
            mu_buf[:, i] = 1./var_t*mu_t*weights[i]

        if self.variance_type == 1:
            return mu, var_1
        elif self.variance_type == 2:
            return mu, var
        else:
            tmp = np.sum(var_buf, axis=1)
            tmp[tmp == 0.] = 1e-5
            var = 1. / tmp
            mu = np.sum(mu_buf, axis=1) * var
            return mu, var

    def optimize(self, pred_y, true_y, pred_y_unlabeled):
        if self._rank_type == 0:
            x, status = scipy_solve(pred_y, true_y, (self._norm, self._lbd, self._mu),
                                    pred_y_unlabeled, debug=self._debug_mode)
        else:
            x, status = scipy_solve_rank(pred_y, true_y, (self._norm, self._lbd, self._mu, self._rank_type),
                                         pred_y_unlabeled, debug=self._debug_mode)
        if status:
            x[x < 1e-3] = 0.
            self.weights = x

    def set_hp(self, norm, lbd):
        self._norm = norm
        self._lbd = lbd

    def set_rank(self, rank_type):
        self._rank_type = rank_type

    def set_mu(self, mu):
        self._mu = mu

    def get_weights(self):
        return self.weights



import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.utils.scipy_solver_m2 import scipy_solve
from tlbo.utils.normalization import zero_one_normalization

'''
This surrogate try to learn the basic surrogates with the auxiliary surrogate.
'''


class OwesomeSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern',
                 loss_type=3, v_type=3, lbd=.3):
        BaseSurrogate.__init__(self, train_metadata, test_metadata, cov_amp=cov_amp, kernel_type=kernel_type,
                               normalize_output=False)
        # Initialize weights for all source surrogates.
        self.weights = np.array([1./self.historical_task_num]*(self.historical_task_num+1))
        self.variance_type = v_type
        self.loss_type = loss_type
        self.lbd = lbd
        # Train each source surrogate.
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
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

    def create_stale_model(self, X, y):
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)
        stale_model = self.create_single_gp(lower, upper)
        instance_num = len(y)
        # Conduct 5-fold evaluation.
        mu, var = np.zeros(instance_num), np.zeros(instance_num)
        K = 3
        if instance_num >= K:
            fold_num = instance_num // K
            for i in range(K):
                row_indexs = list(range(instance_num))
                bound = (instance_num - i * fold_num) if i == (K - 1) else fold_num
                for index in range(bound):
                    del row_indexs[i * fold_num]

                if (y[row_indexs] == y[row_indexs[0]]).all():
                    y[row_indexs[0]] += 1e-5
                do_optimize = True if i == 0 else False
                stale_model.train(X[row_indexs, :], y[row_indexs], do_optimize=do_optimize)
                index_s, index_e = i * fold_num, i * fold_num + bound
                mu_t, var_t = stale_model.predict(X[index_s:index_e, :])
                mu[index_s: index_e] = mu_t
                var[index_s:index_e] = var_t
        return mu, var

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        y, _, _ = zero_one_normalization(y)
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)
        self.current_model = self.create_single_gp(lower, upper)
        self.current_model.train(X, y)

        # Predict the results on labeled data.
        pred_y = self.batch_predict(X)

        mu_t, _ = self.create_stale_model(X, y)
        pred_y = np.r_[pred_y, mu_t.reshape((1, -1))]

        # Learn the weights.
        self.optimize(np.mat(pred_y).T, np.mat(y).T)

    def predict(self, X: np.array):
        n = X.shape[0]
        m = self.weights.shape[0]
        mu, var = np.zeros(n), np.zeros(n)
        weights = self.weights
        max_index = np.argmax(self.weights)
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))

        # Prediction from current surrogate.
        mu_t, var_t = self.current_model.predict(X)
        var_buf[:, -1] = 1. / var_t * weights[-1]
        mu_buf[:, -1] = 1. / var_t * mu_t * weights[-1]
        var = var_t
        var_1 = var_t

        # Predictions from basic surrogates.
        for i in range(0, self.historical_task_num):
            mu_t, var_t = self.historical_model[i].predict(X)
            mu += weights[i] * mu_t
            var += weights[i] * weights[i] * var_t
            if i == max_index:
                var_1 = var_t
            # compute the gaussian experts.
            var_buf[:, i] = 1. / var_t * weights[i]
            mu_buf[:, i] = 1. / var_t * mu_t * weights[i]

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

    def optimize(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, self.lbd, self.loss_type, debug=self._debug_mode)
        if status:
            x[x < 1e-3] = 0.
            self.weights = x

    def get_weights(self):
        return self.weights

    def set_lbd(self, lbd):
        self.lbd = lbd

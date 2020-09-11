import numpy as np
from rtl.facade.base_surrogate import BaseSurrogate
from rtl.utils.scipy_solver_m2 import scipy_solve
from rtl.utils.normalization import zero_one_normalization


class GreatSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern',
                 loss_type=3, v_type=3, lbd=1., use_hedge=False):
        BaseSurrogate.__init__(self, train_metadata, test_metadata, cov_amp=cov_amp, kernel_type=kernel_type,
                               normalize_output=False)
        # Initialize weights for all source surrogates.
        # self.weights = np.array([1./self.historical_task_num]*(self.historical_task_num+1))
        self.weights = np.array([1. / self.historical_task_num] * self.historical_task_num)
        self.variance_type = v_type
        self.loss_type = loss_type
        self.lbd = lbd
        self.alpha = 1.
        self.reduction_rate = 0.3
        self.use_dynamic_lbd = True
        self.duration = 5
        self.train_size = 0
        self.generalization_loss = list()
        self.prior_size = self.train_metadata[0][:, 0].shape[0]
        # hedge algorithm.
        self.init_flag = True
        self.threshold = 1/256
        self.use_hedge = use_hedge
        self.w = np.array([1-self.threshold, self.threshold])
        self.beta = 0.5
        self.p = np.array([1., 0.])

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

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)
        self.train_size = y.shape[0]
        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_one_normalization(y)

        # Using the incoming instance to adjust the lambda.
        if self.variance_type != 3 and X.shape[0] == 3:
            # Train a surrogate on the target problem.
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            self.current_model = self.create_single_gp(lower, upper)
            self.current_model.train(X, y)

        pred_y, _ = self.predict(X, use_fusion=False)
        if self.use_dynamic_lbd and self.lbd > 0.:
            self.adjust_lbd(pred_y, y)
        if self.use_hedge and X.shape[0] != 3:
            pred_ft, _ = self.current_model.predict(X)
            self.adjust_hedge(pred_y, pred_ft, y)

        # Train a surrogate on the target problem.
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)
        self.current_model = self.create_single_gp(lower, upper)
        self.current_model.train(X, y)

        pred_y = self.batch_predict(X)

        # Learn the weights.
        self.optimize(np.mat(pred_y).T, np.mat(y).T)

    # Compute the ranking loss.
    def ranking_loss(self, y_true, y_pred):
        n_sample = y_true.shape[0]
        pairs = list()
        for i in range(n_sample):
            if y_true[-1] > y_true[i]:
                pairs.append((-1, i))
            elif y_true[-1] < y_true[i]:
                pairs.append((i, -1))
        loss = 0
        for (i, j) in pairs:
            loss += np.log(1 + np.exp(y_pred[j] - y_pred[i]))
        if len(pairs) != 0:
            loss = loss / len(pairs)
        return loss

    def adjust_hedge(self, pred_fs, pred_ft, true_y):
        loss_fs = self.ranking_loss(true_y, pred_fs)
        loss_ft = self.ranking_loss(true_y, pred_ft)
        # print(loss_fs, loss_ft)
        if loss_fs < loss_ft:
            self.w[1] *= self.beta
        if loss_ft < loss_fs:
            self.w[0] *= self.beta
            self.init_flag = False
        # Keep the two weight in a margin.
        bound = np.max(self.w)*self.threshold
        self.w[self.w < bound] = bound
        if np.max(self.w) < 1e-5:
            self.w *= 1e5

        # Normalize w to obtain the p.
        if true_y.shape[0] > 3 and not self.init_flag:
            self.p = self.w/np.sum(self.w)
        print(self.p, self.w)

    def adjust_lbd(self, y_pred, y_list):
        n_sample = y_pred.shape[0]
        pairs = list()
        for i in range(n_sample):
            if y_list[-1] > y_list[i]:
                pairs.append((-1, i))
            elif y_list[-1] < y_list[i]:
                pairs.append((i, -1))

        loss = 0
        for (i, j) in pairs:
            loss += np.log(1 + np.exp(y_pred[j] - y_pred[i]))
        if len(pairs) != 0:
            loss = loss/len(pairs)

        if len(self.generalization_loss) >= self.duration:
            if np.max(self.generalization_loss[-self.duration:]) <= loss:
                self.lbd *= self.reduction_rate
                if self.lbd < 1e-3:
                    self.lbd = 0.
                print('Lambda Adjust', self.lbd, y_list.shape[0])
        self.generalization_loss.append(loss)

    def predict(self, X: np.array, use_fusion=True):
        n = X.shape[0]
        m = self.weights.shape[0]
        weights = self.weights
        if self.variance_type == 3:
            var_buf = np.zeros((n, m))
            mu_buf = np.zeros((n, m))
            # Predictions from basic surrogates.
            for i in range(0, self.historical_task_num):
                mu_t, var_t = self.historical_model[i].predict(X)
                # compute the gaussian experts.
                var_buf[:, i] = 1./var_t*weights[i]
                mu_buf[:, i] = 1./var_t*mu_t*weights[i]

            var = 1. / np.sum(var_buf, axis=1)
            mu = np.sum(mu_buf, axis=1) * var

            # Combine the prediction from basic surrogate.
            if use_fusion and self.train_size >= 0:
                mu_target, var_target = self.current_model.predict(X)
                # Compute the proportion.
                if not self.use_hedge:
                    weight_target = self.train_size*self.alpha / (self.train_size*self.alpha + self.prior_size)
                else:
                    # Weight obtained using hedge algorithm.
                    weight_target = self.p[1]

                var_buf = np.zeros((n, 2))
                mu_buf = np.zeros((n, 2))
                mu_buf[:, 0] = 1./var*mu*(1-weight_target)
                var_buf[:, 0] = 1./var*(1-weight_target)
                # target surrogate prediction.
                mu_buf[:, 1] = 1./var_target*mu_target*weight_target
                var_buf[:, 1] = 1./var_target*weight_target
                var = 1. / np.sum(var_buf, axis=1)
                mu = np.sum(mu_buf, axis=1) * var
            return mu, var
        else:
            mu, var = np.zeros(n), np.zeros(n)
            # Prediction from current surrogate.
            mu_t, var_t = self.current_model.predict(X)
            mu += weights[-1]*mu_t
            var += weights[-1]*weights[-1]*var_t
            var_2 = var_t

            # Predictions from basic surrogates.
            for i in range(0, self.historical_task_num):
                mu_t, var_t = self.historical_model[i].predict(X)
                mu += weights[i] * mu_t
                var += weights[i] * weights[i] * var_t
            if self.variance_type == 1:
                return mu, var
            elif self.variance_type == 2:
                return mu, var_2

    def optimize(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, self.lbd, self.loss_type, debug=self._debug_mode)
        if status:
            x[x < 1e-3] = 0.
            self.weights = x

    def get_weights(self):
        return self.weights

    def set_lbd(self, mode, lbd):
        self.lbd = lbd
        self.use_dynamic_lbd = mode

    def set_params(self, alpha, rate):
        self.alpha = alpha
        self.reduction_rate = rate

    def get_p(self):
        return self.p

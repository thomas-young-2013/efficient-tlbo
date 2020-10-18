import numpy as np
from sklearn.model_selection import KFold
from tlbo.facade.base_facade import BaseFacade
from tlbo.utils.scipy_solver import scipy_solve

_scale_method = 'standardize'


class OBTL(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, fusion_method='idp_lc'):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'obtl'
        self.fusion_method = fusion_method
        self.build_source_surrogates(normalize=_scale_method)
        # Weights for base surrogates and the target surrogate.
        self.w = np.array([1. / self.K] * self.K + [0.])
        self.base_predictions = list()
        self.min_num_y = 5
        self.ensemble_size = 50

        self.hist_ws = list()
        self.iteration_id = 0
        self.target_y_range = None

    def batch_predict(self, X: np.ndarray):
        pred_y = None
        for i in range(0, self.K):
            mu, _ = self.source_surrogates[i].predict(X)
            if pred_y is not None:
                pred_y = np.c_[pred_y, mu]
            else:
                pred_y = mu
        return pred_y

    def train(self, X: np.ndarray, y: np.array):
        instance_num = X.shape[0]
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize=_scale_method)
        self.target_y_range = 0.5 * (np.max(y) - np.min(y))
        # print('Target y range', self.target_y_range)
        pred_y = self.batch_predict(X)

        # Learn the weights of source problems.
        self.learn_source_weights(np.mat(pred_y), np.mat(y).T)
        w_source = self.w[:self.K]
        w_target = 0.

        if instance_num >= self.min_num_y:
            w_target = self.calculate_weight_by_sampling(X, y)
            if instance_num >= 2 * self.min_num_y:
                w_target = np.max([w_target, self.w[-1]])
            w_source *= (1 - w_target)
        w_new = np.asarray(list(w_source) + [w_target])
        rho = 0.6
        self.w = rho * w_new + (1 - rho) * self.w

        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        print('In iter-%d' % self.iteration_id)
        print(weight_str)
        self.hist_ws.append(w)
        self.iteration_id += 1

    def learn_source_weights(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, 3, debug=True)
        if status:
            x[x < 1e-3] = 0.
            self.w[:self.K] = x

    def predict_target_surrogate_cv(self, X, y):
        k_fold_num = 5
        _mu, _var = list(), list()

        # Conduct K-fold cross validation.
        kf = KFold(n_splits=k_fold_num)
        idxs = list()
        for train_idx, val_idx in kf.split(X):
            idxs.extend(list(val_idx))
            X_train, X_val, y_train, y_val = X[train_idx,:], X[val_idx,:], y[train_idx], y[val_idx]
            model = self.build_single_surrogate(X_train, y_train, normalize=_scale_method)
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            _mu.extend(list(mu))
            _var.extend(list(var))
        assert (np.array(idxs) == np.arange(X.shape[0])).all()
        return np.asarray(_mu), np.asarray(_var)

    def calculate_weight_by_sampling(self, X, y):
        surrogate_ids = list()
        surrogate_preds = list()
        for idx in range(self.K):
            surrogate_ids.append(idx)
            _mu, _var = self.source_surrogates[idx].predict(X)
            surrogate_preds.append((_mu.flatten(), _var.flatten()))
        target_surrogate_pred = self.predict_target_surrogate_cv(X, y)
        surrogate_ids.append(self.K)
        surrogate_preds.append(target_surrogate_pred)

        surrogate_idx = list()
        _K = len(surrogate_ids)

        for _ in range(self.ensemble_size):
            loss_list = list()
            for i in range(_K):
                _mu, _var = surrogate_preds[i]
                y_pred = np.random.normal(_mu, _var)
                loss_list.append(self.calculate_ranking_loss(y_pred, y))
            argmin_idx = np.argmin(loss_list)
            surrogate_idx.append(argmin_idx)

        _w = np.zeros(_K)
        for idx in surrogate_idx:
            _w[idx] += 1
        _w = _w/np.sum(_w)
        return _w[-1]

    def predict(self, X: np.array):
        return self.combine_predictions(X, self.fusion_method)

    def get_weights(self):
        return self.w

    @staticmethod
    def penalty_func(x1, x2, y1, y2):
        if x1 < x2:
            z = y2 - y1
        else:
            z = y1 - y2
        z *= 10.
        return np.log(1 + np.exp(-z))

    @staticmethod
    def calculate_ranking_loss(y_pred: np.array, y: np.array):
        n = y_pred.shape[0]
        ranking_loss = 0.
        for i in range(n):
            for j in range(i+1, n):
                ranking_loss += OBTL.penalty_func(y[i], y[j], y_pred[i], y_pred[j])
        return ranking_loss / (n * n)

    @staticmethod
    def calculate_generalization_ranking_loss(y_pred: np.array, y: np.array, start_idx=0):
        n = y_pred.shape[0]
        ranking_loss = 0.
        for i in range(start_idx, n):
            for j in range(n):
                ranking_loss += OBTL.penalty_func(y[i], y[j], y_pred[i], y_pred[j])
        return ranking_loss

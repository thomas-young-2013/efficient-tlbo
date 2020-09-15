import numpy as np
from tlbo.facade.base_facade import BaseFacade


class ES(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, rng,
                 surrogate_type='gp', num_src_hpo_trial=50, fusion_method='idp'):
        super().__init__(config_space, source_hpo_data, rng, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.fusion_method = fusion_method
        self.build_source_surrogates()
        # Weights for base surrogates and the target surrogate.
        self.w = np.array([1. / self.K] * self.K + [0.])
        self.ensemble_size = 200
        self.base_predictions = list()
        self.min_num_y = 5

    @staticmethod
    def penalty_func(x1, x2, y1, y2):
        if x1 < x2:
            z = y2 - y1
        else:
            z = y1 - y2
        return np.log(1 + np.exp(-z))
        # return int((x1 < x2) ^ (y1 < y2))

    @staticmethod
    def calculate_ranking_loss(y_pred: np.array, y: np.array):
        n = len(y_pred)
        assert len(y) == n
        ranking_loss = 0.
        for i in range(n):
            for j in range(i+1, n):
                ranking_loss += ES.penalty_func(y[i], y[j], y_pred[i], y_pred[j])
        return ranking_loss / (n * n)

    @staticmethod
    def calculate_generalization_loss(y_pred: np.array, y: np.array):
        n = len(y_pred)
        assert len(y) == n
        ranking_loss = 0.
        j = n - 1
        for i in range(n - 1):
            ranking_loss += ES.penalty_func(y[i], y[j], y_pred[i], y_pred[j])
        return ranking_loss / n

    def train(self, X: np.ndarray, y: np.array):
        # Compute each base surrogate's predictions.
        base_predictions = list()
        for i in range(self.K):
            mean, _ = self.source_surrogates[i].predict(X)
            mean = mean.flatten()
            base_predictions.append(mean)
        self.surrogate_ensemble = list()
        surrogate_idx = list()

        for iter_id in range(self.ensemble_size):
            loss_list = list()
            for i in range(self.K):
                if len(self.surrogate_ensemble) == 0:
                    _predictions = base_predictions[i]
                else:
                    _predictions = np.mean(self.surrogate_ensemble + [base_predictions[i]], axis=0)
                loss_list.append(self.calculate_ranking_loss(_predictions, y))
            argmin_idx = np.argmin(loss_list)
            # min_loss = np.min(loss_list)

            self.surrogate_ensemble.append(base_predictions[argmin_idx])
            surrogate_idx.append(argmin_idx)
            # print('in iter %d' % iter_id, 'loss is %.4f' % min_loss)

        # Update base surrogates' weights.
        w_source = np.zeros(self.K)
        for idx in surrogate_idx:
            w_source[idx] += 1
        w_source = w_source/np.sum(w_source)
        self.w[:-1] = w_source

        if len(y) >= self.min_num_y:
            w1, w2 = self.calculate_target_weight(X, y)
        else:
            w1, w2 = 1., 0.

        self.w[:-1] *= w1
        self.w[-1] = w2

        # Disable the target surrogate.
        # self.w[:-1] = self.w[:-1]/np.sum(self.w[:-1])
        # self.w[-1] = 0.
        # self.w[:-1] = np.zeros(self.K)
        # self.w[-1] = 1.
        print('=' * 20)
        print('current weights', self.w)

        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y)

    def calculate_weight_generalization(self, X, y):
        weight = self.w.copy()
        weight[-1] = 0.
        weight[:-1] = weight[:-1]/np.sum(weight[:-1])
        mu_list, _ = self.combine_predictions(X, combination_method='idp_lc', weight=weight)
        pred_source = mu_list.flatten()
        loss_ss = self.calculate_generalization_loss(pred_source, y)
        pred_target, _ = self.target_surrogate.predict(X)
        pred_target = pred_target.flatten()
        loss_ts = self.calculate_generalization_loss(pred_target, y)
        print(loss_ss, loss_ts)

    def calculate_target_weight(self, X, y):
        source_prediction = np.mean(self.surrogate_ensemble, axis=0)
        if self.target_surrogate is None:
            # print('=' * 20)
            # print('#Iteration=', len(y), 'target surrogate is none!')
            # print('=' * 20)
            return 1., 0.
        target_prediction, _ = self.target_surrogate.predict(X)
        base_predictions = [source_prediction.flatten(), target_prediction.flatten()]

        _ensemble = list()
        surrogate_idx = list()
        _K = 2

        for iter_id in range(self.ensemble_size):
            loss_list = list()
            for i in range(_K):
                _predictions = np.mean(_ensemble + [base_predictions[i]], axis=0)
                loss_list.append(self.calculate_generalization_loss(_predictions, y))
            argmin_idx = np.argmin(loss_list)
            _ensemble.append(base_predictions[argmin_idx])
            surrogate_idx.append(argmin_idx)

        _w = np.zeros(len(base_predictions))
        for idx in surrogate_idx:
            _w[idx] += 1
        _w = _w/np.sum(_w)
        print(_w)
        return _w

    def calculate_weight(self, X, y):
        weight = self.w.copy()
        weight[-1] = 0.
        weight[:-1] = weight[:-1]/np.sum(weight[:-1])
        mu_list, var_list = self.combine_predictions(X, combination_method='idp_lc', weight=weight)
        mu_list, var_list = mu_list.flatten(), var_list.flatten()

        k_fold_num = 5
        cached_mu_list, cached_var_list = list(), list()
        instance_num = len(y)
        skip_target_surrogate = False if instance_num >= k_fold_num else True

        if not skip_target_surrogate:
            # Conduct K-fold cross validation.
            fold_num = instance_num // k_fold_num
            for i in range(k_fold_num):
                row_indexs = list(range(instance_num))
                bound = (instance_num - i * fold_num) if i == (k_fold_num - 1) else fold_num
                for index in range(bound):
                    del row_indexs[i * fold_num]

                if (y[row_indexs] == y[row_indexs[0]]).all():
                    y[row_indexs[0]] += 1e-4
                model = self.build_single_surrogate(X[row_indexs, :], y[row_indexs])
                mu, var = model.predict(X)
                cached_mu_list.append(mu.flatten())
                cached_var_list.append(var.flatten())

        argmin_list = np.zeros(2)
        for _ in range(100):
            ranking_loss_list = list()
            sampled_y = np.random.normal(mu_list, var_list)
            _loss_ss = self.calculate_ranking_loss(sampled_y, y)
            ranking_loss_list.append(_loss_ss)

            # Compute ranking loss for target surrogate.
            _loss_ts = 0
            if not skip_target_surrogate:
                fold_num = instance_num // k_fold_num
                for fold in range(k_fold_num):
                    sampled_y = np.random.normal(cached_mu_list[fold], cached_var_list[fold])
                    bound = instance_num if fold == (k_fold_num - 1) else (fold + 1) * fold_num
                    for i in range(fold_num * fold, bound):
                        for j in range(instance_num):
                            _loss_ts += self.penalty_func(y[i], y[j], sampled_y[i], sampled_y[j])
            else:
                _loss_ts = instance_num * instance_num
            ranking_loss_list.append(_loss_ts / (instance_num * instance_num))
            print(ranking_loss_list)
            argmin_task = np.argmin(ranking_loss_list)
            argmin_list[argmin_task] += 1

        # Update the weights.
        w1, w2 = argmin_list / np.sum(argmin_list)
        print(w1, w2)
        return w1, w2

    def predict(self, X: np.array):
        return self.combine_predictions(X, combination_method='idp_lc')

    def get_weights(self):
        return self.w

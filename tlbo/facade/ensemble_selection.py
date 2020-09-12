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
        self.ensemble_size = 100
        self.base_predictions = list()
        self.min_num_y = 5

    @staticmethod
    def calculate_ranking_loss(y_pred: np.array, y: np.array):
        n = len(y_pred)
        assert len(y) == n
        ranking_loss = 0.
        for i in range(n):
            for j in range(i+1, n):
                if y[i] < y[j]:
                    z = y_pred[j] - y_pred[i]
                elif y[j] < y[i]:
                    z = y_pred[i] - y_pred[j]
                else:
                    continue
                ranking_loss += np.log(1 + np.exp(-z))
        return ranking_loss / (n * n)

    @staticmethod
    def calculate_generalization_loss(y_pred: np.array, y: np.array):
        n = len(y_pred)
        assert len(y) == n
        ranking_loss = 0.
        j = n - 1
        for i in range(n - 1):
            if y[i] < y[j]:
                z = y_pred[j] - y_pred[i]
            elif y[j] < y[i]:
                z = y_pred[i] - y_pred[j]
            else:
                continue
            ranking_loss += np.log(1 + np.exp(-z))
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

        if len(y) > self.min_num_y:
            w1, w2 = self.calculate_target_weight(X, y)
        else:
            w1, w2 = 1., 0.

        self.w[:-1] *= w1
        self.w[-1] = w2
        # print('=' * 20)
        # print('current weights', self.w)

        # Build the target surrogate.
        if len(y) >= self.min_num_y - 1:
            self.target_surrogate = self.build_single_surrogate(X, y)

    def calculate_target_weight(self, X, y):
        source_prediction = np.mean(self.surrogate_ensemble, axis=0)
        if self.target_surrogate is None:
            print('=' * 20)
            print('#Iteration=', len(y), 'target surrrogate is none!')
            print('=' * 20)
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
        return _w

    def predict(self, X: np.array):
        if self.target_surrogate is None:
            mu, var = np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))
        else:
            mu, var = self.target_surrogate.predict(X)

        # Target surrogate predictions with weight.
        mu *= self.w[-1]
        var *= (self.w[-1] * self.w[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.K):
            mu_t, var_t = self.source_surrogates[i].predict(X)
            mu += self.w[i] * mu_t
            var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_weights(self):
        return self.w

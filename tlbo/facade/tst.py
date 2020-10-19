import numpy as np
from tlbo.facade.base_facade import BaseFacade


class TST(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, use_metafeatures=False, metafeatures=None, only_source=False):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'tst'
        self.only_source = only_source
        self.build_source_surrogates(normalize='scale')
        # Weights for base surrogates and the target surrogate.
        self.w = [0.75] * (self.K + 1)
        self.use_metafeatures = use_metafeatures
        self.metafeatures = metafeatures
        self.bandwidth = 0.45
        if self.use_metafeatures:
            assert len(metafeatures) == self.K + 1
            self.meta_dist = [0.] * self.K
            for _id in range(self.K):
                _dist = np.power(self.metafeatures[self.K] - self.metafeatures[_id], 2)
                self.meta_dist = _dist

        # Preventing weight dilution.
        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='scale')

        n_sample = X.shape[0]
        for _id in range(self.K):
            if not self.use_metafeatures:
                mu, _ = self.source_surrogates[_id].predict(X)
                discordant_paris, total_pairs = 0, 0
                for i in range(n_sample):
                    for j in range(i + 1, n_sample):
                        if (y[i] < y[j]) ^ (mu[i] < mu[j]):
                            discordant_paris += 1
                        total_pairs += 1
                tmp = discordant_paris / total_pairs / self.bandwidth
            else:
                tmp = self.meta_dist[_id] / self.bandwidth
            print(tmp)
            self.w[_id] = 0.75 * (1 - tmp * tmp) if tmp <= 1 else 0

        if self.only_source:
            self.w[-1] = 0.
            self.w = np.array(self.w) / np.sum(self.w)

        print('=' * 20)
        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        print('In iter-%d' % self.iteration_id)

        self.target_weight.append(w[-1] / sum(w))
        print(weight_str)
        self.hist_ws.append(w)
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        denominator = 0.75
        for i in range(0, self.K):
            weight = self.w[i]
            mu_t, _ = self.source_surrogates[i].predict(X)
            mu += weight * mu_t
            denominator += weight
        mu /= denominator
        return mu, var

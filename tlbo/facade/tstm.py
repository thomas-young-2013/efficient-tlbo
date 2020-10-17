import numpy as np
from tlbo.facade.base_facade import BaseFacade


class TSTM(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, metafeatures=None):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'tst'
        self.build_source_surrogates(normalize='scale')
        # Weights for base surrogates and the target surrogate.
        self.w = [0.75] * (self.K + 1)

        self.metafeatures = np.zeros(shape=(len(metafeatures), len(metafeatures[0])))
        self.metafeatures[:-1] = self.scale_fit_meta_features(metafeatures[:-1])
        self.metafeatures[-1] = self.scale_transform_meta_features(metafeatures[-1])

        self.bandwidth = 1.8

        assert len(metafeatures) == self.K + 1
        self.meta_dist = [0.] * self.K

        for _id in range(self.K):
            _dist = np.sqrt(np.sum(np.square(self.metafeatures[self.K] - self.metafeatures[_id])))
            self.meta_dist[_id] = _dist

        # Preventing weight dilution.
        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='scale')

        for _id in range(self.K):
            tmp = self.meta_dist[_id] / self.bandwidth
            print(tmp)
            self.w[_id] = 0.75 * (1 - tmp * tmp) if tmp <= 1 else 0

        print('=' * 20)
        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        print('In iter-%d' % self.iteration_id)
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

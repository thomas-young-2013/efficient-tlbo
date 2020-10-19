import numpy as np
from tlbo.facade.base_facade import BaseFacade


class POGPE(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'pogpe'
        self.only_source = only_source
        self.build_source_surrogates(normalize='scale')

        self.w = np.array([0.5/self.K] * self.K + [0.5])
        # Preventing weight dilution.
        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='scale')

        print('=' * 20)
        if self.only_source:
            self.w[-1] = 0.
            self.w = self.w / np.sum(self.w)

        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        print('In iter-%d' % self.iteration_id)
        print(weight_str)
        self.hist_ws.append(w)
        self.iteration_id += 1

    def predict(self, X: np.array):
        w = self.w.copy()
        n, m = X.shape[0], len(self.w)
        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))

        # Predictions from source surrogates.
        for i in range(0, self.K + 1):
            if i == self.K:
                if self.target_surrogate is not None:
                    _mu, _var = self.target_surrogate.predict(X)
                else:
                    _mu, _var = 0., 0.
            else:
                _mu, _var = self.source_surrogates[i].predict(X)

            _mu, _var = _mu.flatten(), _var.flatten()
            if (_var != 0).all():
                var_buf[:, i] = (1./_var*w[i])
                mu_buf[:, i] = (1./_var*_mu*w[i])

        tmp = np.sum(var_buf, axis=1)
        tmp[tmp == 0.] = 1e-5
        var = 1. / tmp
        mu = np.sum(mu_buf, axis=1) * var
        return mu.reshape(-1, 1), var.reshape(-1, 1)

import numpy as np
from tlbo.facade.norm import NORM


class NORMTST(NORM):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed, norm=3,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, target_hp_configs, seed, norm,
                         surrogate_type, num_src_hpo_trial, only_source)

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        denominator = 0.75
        for i in range(0, self.K):
            weight = self.sw[i]
            mu_t, _ = self.source_surrogates[i].predict(X)
            mu += weight * mu_t
            denominator += weight
        mu /= denominator
        return mu, var

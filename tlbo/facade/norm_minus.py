import numpy as np
from tlbo.facade.norm import NORM


class NORMMinus(NORM):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed, norm=3,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, target_hp_configs, seed, norm,
                         surrogate_type, num_src_hpo_trial, only_source)

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        # # Target surrogate predictions with weight.
        # mu *= self.w[-1]
        # var *= (self.w[-1] * self.w[-1])
        #
        # # Base surrogate predictions with corresponding weights.
        # for i in range(0, self.K):
        #     if not self.ignored_flag[i]:
        #         mu_t, var_t = self.source_surrogates[i].predict(X)
        #         mu += self.w[i] * mu_t
        #         var += self.w[i] * self.w[i] * var_t
        return mu, var

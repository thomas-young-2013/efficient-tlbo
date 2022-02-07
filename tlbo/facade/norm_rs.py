import numpy as np
from tlbo.facade.norm import NORM


class NORMRandomSearch(NORM):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed, norm=3,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, target_hp_configs, seed, norm,
                         surrogate_type, num_src_hpo_trial, only_source)
        self.rng = np.random.RandomState(seed)

    def predict(self, X: np.array):
        # Imitate the random search.
        n = X.shape[0]
        return self.rng.rand(n, 1), np.array([1e-5] * n).reshape(-1, 1)

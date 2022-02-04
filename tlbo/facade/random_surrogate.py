import numpy as np
from tlbo.facade.base_facade import BaseFacade


class RandomSearch(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='gp', num_src_hpo_trial=50, fusion_method='idp'):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'rs'
        self.rng = np.random.RandomState(seed)

    def train(self, X: np.ndarray, y: np.array):
        pass

    def predict(self, X: np.array):
        # Imitate the random search.
        n = X.shape[0]
        return self.rng.rand(n, 1), np.array([1e-5]*n).reshape(-1, 1)

import numpy as np
from tlbo.facade.base_facade import BaseFacade


class NoTL(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'notl'

    def train(self, X: np.ndarray, y: np.array):
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='standardize')

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        return mu, var

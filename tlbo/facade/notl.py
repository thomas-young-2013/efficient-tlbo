import numpy as np
from tlbo.facade.base_facade import BaseFacade


class NoTL(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, rng):
        super().__init__(config_space, source_hpo_data, rng, target_hp_configs)

    def train(self, X: np.ndarray, y: np.array):
        self.target_surrogate = self.build_single_surrogate(X, y)

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        return mu, var

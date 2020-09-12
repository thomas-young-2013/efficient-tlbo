import abc
import typing
import numpy as np
from typing import List

from tlbo.model.model_builder import build_model
from tlbo.config_space import ConfigurationSpace, Configuration
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class BaseFacade(object):
    def __init__(self, config_space: ConfigurationSpace,
                 source_hpo_data: List,
                 target_hp_configs: List,
                 rng: np.random.RandomState,
                 history_dataset_features: List = None,
                 surrogate_type='gp_mcmc'):
        self.config_space = config_space
        self.rng = rng
        # The number of source problems.
        self.K = len(source_hpo_data)
        self.source_hpo_data = source_hpo_data
        self.target_hp_configs = target_hp_configs
        self.source_surrogates = None
        self.target_surrogate = None
        self.history_dataset_features = history_dataset_features
        if history_dataset_features is not None:
            assert len(history_dataset_features) == self.K
        self.surrogate_type = surrogate_type

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def build_source_surrogates(self):
        self.source_surrogates = list()
        for hpo_evaluation_data in self.source_hpo_data:
            model = build_model(self.surrogate_type, self.config_space, self.rng)
            _X, _y = list(), list()
            for _config, _config_perf in hpo_evaluation_data.items():
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)

            # Prevent the same value in y.
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
            model.train(X, y)
            self.source_surrogates.append(model)

    def build_single_surrogate(self, X: np.ndarray, y: np.array):
        model = build_model(self.surrogate_type, self.config_space, self.rng)
        # Prevent the same value in y.
        if (y == y[0]).all():
            y[0] += 1e-4
        y, _, _ = zero_mean_unit_var_normalization(y)
        model.train(X, y)
        return model

from typing import List, Union, Tuple
import numpy as np

from tlbo.acquisition_function.acquisition import AbstractAcquisitionFunction
from tlbo.config_space import Configuration, ConfigurationSpace
from tlbo.utils.history_container import HistoryContainer
from tlbo.optimizer.ei_optimization import AcquisitionFunctionMaximizer


class OfflineSearch(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            configuration_list: List[Configuration],
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None):
        super().__init__(acquisition_function, config_space, rng)
        self.configuration_list = configuration_list

    def _maximize(
            self,
            runhistory: HistoryContainer,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        _configs = self.configuration_list
        for i in range(len(_configs)):
            _configs[i].origin = 'Offline Search (sorted)'
        return self._sort_configs_by_acq_value(_configs)

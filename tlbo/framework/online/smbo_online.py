import time
import numpy as np
import traceback
from typing import List, Dict
from tlbo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch
from tlbo.utils.constants import MAXINT, SUCCESS, FAILDED

from tlbo.framework.smbo_offline import SMBO_OFFLINE


class SMBO_ONLINE(SMBO_OFFLINE):
    def __init__(self,
                 objective_function,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.objective_function = objective_function
        self.acq_optimizer = InterleavedLocalAndRandomSearch(   # replace offline search
            self.acquisition_function,
            self.config_space,
            rng=np.random.RandomState(self.random_seed),
            max_steps=None,
            n_steps_plateau_walk=10,
            n_sls_iterations=10,
            rand_prob=0.0,
        )

    def evaluate(self, config):
        try:
            perf = self.target_hpo_measurements.get(config)
            if perf is None:
                print('[Evaluation] on real objective!')
                perf = self.objective_function(config)
            else:
                print('[Evaluation] get cache.')
        except Exception:
            print(traceback.format_exc())
            perf = MAXINT
        return perf

    def sample_random_config(self, config_num=1):
        configs = list()
        sample_cnt = 0
        while len(configs) < config_num:
            sample_cnt += 1
            config = self.config_space.sample_configuration(1)
            if config not in (self.configurations + self.failed_configurations + configs):
                configs.append(config)
                sample_cnt = 0
            else:
                sample_cnt += 1
            if sample_cnt >= 200:
                configs.append(config)
                sample_cnt = 0
        return configs

    def choose_next(self, *args, **kwargs):
        try:
            return super().choose_next(*args, **kwargs)
        except ValueError:
            print('Error in choose_next online.')
            raise

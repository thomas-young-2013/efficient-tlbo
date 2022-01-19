import time
import traceback

import numpy as np
from typing import List, Dict
from tlbo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch
from tlbo.utils.constants import MAXINT, SUCCESS, FAILDED

from tlbo.framework.smbo_sst import SMBO_SEARCH_SPACE_TRANSFER


class SMBO_SEARCH_SPACE_TRANSFER_ONLINE(SMBO_SEARCH_SPACE_TRANSFER):
    def __init__(self,
                 objective_function,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.objective_function = objective_function
        self.online_acq_optimizer = InterleavedLocalAndRandomSearch(   # replace offline search
            self.acquisition_function,
            self.config_space,
            rng=self.rng,
            max_steps=None,
            n_steps_plateau_walk=10,
            n_sls_iterations=10,
            rand_prob=0.0,
        )
        self.configuration_list_copy = self.configuration_list.copy()

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

    def update_configuration_list(self):
        """
        start offline search based on online searched samples
        """
        start_time = time.time()
        challengers = self.online_acq_optimizer.maximize(
            runhistory=self.history_container,
            num_points=5000,
        )
        self.configuration_list = challengers.challengers + self.configuration_list_copy
        print('optimizing online acq func took', time.time() - start_time)

    def choose_config_target_space(self):
        return super().choose_config_target_space()

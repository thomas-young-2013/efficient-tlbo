import time
import random
import numpy as np
from tlbo.facade.nn_smfo_surrogate import NNSMFO
from tlbo.facade.random_surrogate import RandomSearch


class SMBO(object):
    def __init__(self, metadata, acquisition_func, surrogate, num_initial_design=2, random_seed=0):
        self.seed = random_seed
        self.metadata = metadata
        self.acquisition_func = acquisition_func
        self.surrogate = surrogate

        f = self.metadata[:, 0]
        self.f_max = max(f)
        self.f_min = min(f)

        self.candidates = list()
        self.X = list()
        self.y = list()

        self.iterate_id = 0
        self.num_initial_design = num_initial_design
        self.best_historic_configs = None
        self.time_cost = list()
        self.best_x = None
        self.best_y = None
        self.init_perf = None
        for item in metadata:
            self.candidates.append(item)
        # Detect the surrogate type.
        if isinstance(self.surrogate, NNSMFO):
            self.model_type = 1
        elif isinstance(self.surrogate, RandomSearch):
            self.model_type = 2
        else:
            self.model_type = 0

    def choose_next_config(self):
        # max_ei = -1
        # best_x = list()
        # for index, item in enumerate(self.candidates):
        #     if self.model_type == 0:
        #         ei = self.acquisition_func.compute(np.array([item[1:]]))
        #     elif self.model_type == 1:
        #         avg_rank = self.surrogate.predict(np.array([item[1:]]))
        #         ei = avg_rank
        #     else:
        #         raise ValueError('Invalid model type!')
        #     if ei > max_ei:
        #         best_x.clear()
        #         max_ei = ei
        #         best_x.append((item, index))
        #     elif ei == max_ei:
        #         best_x.append((item, index))
        # return random.choice(best_x)
        if self.model_type == 0:
            try:
                ei = self.acquisition_func.compute(np.array(self.candidates)[:, 1:])
            except IndexError as e:
                print(self.candidates)
                print(len(self.candidates))
                print(e)
        elif self.model_type == 1:
            avg_rank = self.surrogate.predict(np.array(self.candidates)[:, 1:])
            ei = avg_rank
        elif self.model_type == 2:
            ei = np.random.rand(len(self.candidates))
        else:
            raise ValueError('Invalid model type!')
        index = np.argmax(ei)
        config = self.candidates[index]
        # print(config, index)
        return config, index

    def iterate(self):
        if len(self.X) < 2:
            self.init_design()
        time_tick = time.clock()
        self.surrogate.train(np.array(self.X), np.array(self.y))
        x, index = self.choose_next_config()

        x = x[1:]
        y = self.candidates[index][0]
        del self.candidates[index]
        self.update_incumbent(x, y)
        self.time_cost.append(time.clock() - time_tick)

    def get_best_accuracy(self):
        assert self.f_max != self.f_min
        return (self.best_y-self.f_min) / (self.f_max - self.f_min)

    def get_relative_error(self):
        assert self.f_max != self.f_min
        return (self.f_max - self.best_y) / (self.f_max - self.f_min)

    def get_time_cost(self):
        return self.time_cost

    def update_incumbent(self, x, y):
        if self.best_x is None or self.best_y < y:
            self.best_y = y
            self.best_x = x
        self.X.append(x)
        self.y.append(y)
        self.iterate_id += 1

    def index_candidates(self, config):
        index, y = -1, -1
        for i, item in enumerate(self.candidates):
            if (item[1:] == config).all():
                index, y = i, item[0]
        assert index != -1
        return index, y

    def init_design(self):
        if self.surrogate.init != -1:
            time_tick = time.clock()
            configs = self.surrogate.choose_ws_configurations(self.num_initial_design)
            avg_time_tick = (time.clock() - time_tick)/len(configs)
            acc_list, err_list = list(), list()
            for config in configs:
                index, y = self.index_candidates(config)
                del self.candidates[index]
                self.update_incumbent(config, y)
                acc_list.append(self.get_best_accuracy())
                err_list.append(self.get_relative_error())
                self.time_cost.append(avg_time_tick)
            self.init_perf = (acc_list, err_list)
        else:
            random.seed(self.seed)
            for _ in range(self.num_initial_design):
                index = random.randint(0, len(self.candidates)-1)
                x, y = self.candidates[index][1:], self.candidates[index][0]
                del self.candidates[index]
                self.update_incumbent(x, y)
                self.time_cost.append(0.)

    def get_initial_perf(self):
        return self.init_perf

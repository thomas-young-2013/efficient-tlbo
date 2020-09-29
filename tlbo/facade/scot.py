import numpy as np
from tlbo.utils.rank_svm import RankSVM
from tlbo.facade.base_facade import BaseFacade
from tlbo.config_space.util import convert_configurations_to_array


class SCoT(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, metafeatures=None):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'scot'
        self.metafeatures = metafeatures
        if metafeatures is None:
            raise ValueError('SCoT needs meta-features about the datasets.')
        else:
            assert len(metafeatures) == (self.K + 1)

        self.X, self.y = None, None
        for i, hpo_evaluation_data in enumerate(self.source_hpo_data):
            _X, _y = list(), list()
            for _config, _config_perf in hpo_evaluation_data.items():
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)
            X = X[:self.num_src_hpo_trial]
            y = y[:self.num_src_hpo_trial]
            meta_vec = np.array([list(metafeatures[i]) for _ in range(len(y))])
            X = np.c_[X, meta_vec]

            num_sample = X.shape[0]
            y = np.c_[y, np.array([i] * num_sample)]
            if self.X is not None:
                self.X = np.r_[self.X, X]
                self.y = np.r_[self.y, y]
            else:
                self.X, self.y = X, y
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        num_sample = y.shape[0]
        meta_vec = np.array([list(self.metafeatures[self.K]) for _ in range(num_sample)])

        _X = np.c_[X, meta_vec]
        _X = np.r_[self.X, _X]
        _y = np.c_[y, np.array([self.K] * num_sample)]
        _y = np.r_[self.y, y]
        rank_svm = RankSVM()
        rank_svm.fit(_X, _y)
        pred_y = rank_svm.predict(_X)
        self.target_surrogate = self.build_single_surrogate(_X, pred_y, normalize='none')
        self.iteration_id += 1

    def predict(self, X: np.array):
        num_sample = X.shape[0]
        meta_vec = np.array([list(self.metafeatures[self.K]) for _ in range(num_sample)])
        _X = np.c_[X, meta_vec]
        mu, var = self.target_surrogate.predict(_X)
        return mu, var

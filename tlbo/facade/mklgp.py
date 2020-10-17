import numpy as np
from tlbo.facade.base_facade import BaseFacade
from tlbo.model.mkl_gp import MKLGaussianProcess
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class MKLGP(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, metafeatures=None):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'mklgp'
        if metafeatures is None:
            raise ValueError('SCoT needs meta-features about the datasets.')
        else:
            assert len(metafeatures) == (self.K + 1)

        self.metafeatures = np.zeros(shape=(len(metafeatures), len(metafeatures[0])))
        self.metafeatures[:-1] = self.scale_fit_meta_features(metafeatures[:-1])
        self.metafeatures[-1] = self.scale_transform_meta_features(metafeatures[-1])

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
            if (y == y[0]).all():
                y[0] += 1e-5
            y, _, _ = zero_mean_unit_var_normalization(y)

            meta_vec = np.array([list(self.metafeatures[i]) for _ in range(len(y))])
            X = np.c_[X, meta_vec]

            if self.X is not None:
                self.X = np.r_[self.X, X]
                self.y = np.r_[self.y, y]
            else:
                self.X, self.y = X, y

        self.target_surrogate = MKLGaussianProcess(self.metafeatures)
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        num_sample = y.shape[0]
        meta_vec = np.array([list(self.metafeatures[self.K]) for _ in range(num_sample)])

        _X = np.c_[X, meta_vec]
        _X = np.r_[self.X, _X]
        _y = y.copy()
        if (_y == _y[0]).all():
            _y[0] += 1e-5
        _y, _, _ = zero_mean_unit_var_normalization(_y)
        _y = np.r_[self.y, _y]

        # use_optimize = True if self.iter%3 == 0 else False
        self.target_surrogate.train(X, y, optimize=True)
        # self.target_surrogate = self.build_single_surrogate(_X, _y, normalize='none')
        self.iteration_id += 1

    def predict(self, X: np.array):
        return self.target_surrogate.predict(X)

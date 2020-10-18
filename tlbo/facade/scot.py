import numpy as np
from tlbo.config_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from tlbo.utils.rank_svm import RankSVM
from tlbo.facade.base_facade import BaseFacade
from tlbo.model.model_builder import build_model
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.normalization import zero_mean_unit_var_normalization, zero_one_normalization


class SCoT(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, metafeatures=None):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'scot'
        if metafeatures is None:
            raise ValueError('SCoT needs meta-features about the datasets.')
        else:
            # Should include the meta-feature for target problem.
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

            meta_vec = np.array([list(self.metafeatures[i]) for _ in range(len(y))])
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
        _y = np.r_[self.y, _y]
        rank_svm = RankSVM()
        rank_svm.fit(_X, _y)
        pred_y = rank_svm.predict(_X)
        print('Rank SVM training finished.')
        self.target_surrogate = self.build_single_surrogate(_X, pred_y, normalize='none')
        self.iteration_id += 1

    def predict(self, X: np.array):
        num_sample = X.shape[0]
        target_meta_feature = self.metafeatures[self.K]
        meta_vec = np.array([list(target_meta_feature) for _ in range(num_sample)])
        _X = np.c_[X, meta_vec]
        mu, var = self.target_surrogate.predict(_X)
        return mu, var

    def build_single_surrogate(self, X: np.ndarray, y: np.array, normalize):
        assert normalize in ['standardize', 'scale', 'none']
        # Construct hyperspace with meta-features.
        config_space = ConfigurationSpace()
        for hp in self.config_space.get_hyperparameters():
            config_space.add_hyperparameter(hp)
        for cond in self.config_space.get_conditions():
            config_space.add_condition(cond)
        for bid in self.config_space.get_forbiddens():
            config_space.add_forbidden_clause(bid)
        _meta_feature_size = X.shape[1] - len(self.config_space.get_hyperparameters())
        for _idx in range(_meta_feature_size):
            _meta_hp = UniformFloatHyperparameter("meta_feature_%d" % _idx, 0., 1., default_value=0., log=False)
            config_space.add_hyperparameter(_meta_hp)

        model = build_model(self.surrogate_type, config_space, np.random.RandomState(self.random_seed))
        if normalize == 'standardize':
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
        elif normalize == 'scale':
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_one_normalization(y)
        else:
            pass

        model.train(X, y)
        return model

import time
import numpy as np
from typing import List, Dict
from tlbo.model.util_funcs import get_rng, get_types
from tlbo.acquisition_function.acquisition import EI
from tlbo.config_space import Configuration, ConfigurationSpace
from tlbo.facade.notl import NoTL
from tlbo.optimizer.ei_offline_optimizer import OfflineSearch
from tlbo.optimizer.random_configuration_chooser import ChooserProb
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.constants import MAXINT, SUCCESS, FAILDED
from tlbo.utils.normalization import zero_mean_unit_var_normalization, zero_one_normalization
from tlbo.acquisition_function.ta_acquisition import TAQ_EI
from tlbo.framework.smbo import BasePipeline
from tlbo.facade.base_facade import BaseFacade


class SMBO_OFFLINE(BasePipeline):
    def __init__(self,
                 target_hpo_data: Dict,
                 config_space: ConfigurationSpace,
                 surrogate_model: BaseFacade,
                 acq_func: str = 'ei',
                 source_hpo_data=None,
                 enable_init_design=False,
                 num_src_hpo_trial=50,
                 surrogate_type='rf',
                 max_runs=200,
                 logging_dir='./logs',
                 initial_runs=3,
                 task_id=None,
                 random_seed=None):
        super().__init__(config_space, task_id, output_dir=logging_dir)
        self.logger = super()._get_logger(self.__class__.__name__)
        if random_seed is None:
            _, rng = get_rng()
            random_seed = rng.randint(MAXINT)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed)
        self.source_hpo_data = source_hpo_data
        self.num_src_hpo_trial = num_src_hpo_trial
        self.surrogate_type = surrogate_type
        self.acq_func = acq_func

        self.max_iterations = max_runs
        self.iteration_id = 0
        self.default_obj_value = MAXINT

        self.target_hpo_measurements = target_hpo_data
        self.configuration_list = list(self.target_hpo_measurements.keys())
        print('Target problem space: %d configurations' % len(self.configuration_list))
        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()

        if enable_init_design:
            self.initial_configurations = self.initial_design(initial_runs)
        else:
            self.initial_configurations = None

        if self.initial_configurations is None:
            self.init_num = initial_runs
        else:
            self.init_num = len(self.initial_configurations)

        # Initialize the basic component in BO.
        self.config_space.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.model = surrogate_model
        if self.acq_func == 'ei':
            self.acquisition_function = EI(self.model)
        elif self.acq_func == 'taf':
            self.acquisition_function = TAQ_EI(self.model.target_surrogate,
                                               self.model.source_surrogates)
            self.acquisition_function.update(source_etas=self.model.eta_list)
        else:
            raise ValueError('invalid acquisition function ~ %s.' % self.acq_func)

        self.acq_optimizer = OfflineSearch(self.configuration_list,
                                           self.acquisition_function,
                                           config_space,
                                           rng=np.random.RandomState(self.random_seed)
                                           )
        self.random_configuration_chooser = ChooserProb(
            prob=0.1,
            rng=np.random.RandomState(self.random_seed)
        )

        # Set the parameter in metric.
        self.ys = list(self.target_hpo_measurements.values())
        self.y_max, self.y_min = np.max(self.ys), np.min(self.ys)

    def get_adtm(self):
        y_inc = self.get_inc_y()
        assert self.y_max != self.y_min
        return (y_inc - self.y_min) / (self.y_max - self.y_min)

    def get_inc_y(self):
        if isinstance(self.model, NoTL):
            _perfs = [_perf for (_, _perf) in list(self.target_hpo_measurements.items())[:self.iteration_id]]
            y_inc = np.min(_perfs)
        else:
            y_inc = np.min(self.perfs)
        return y_inc

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def sort_configs_by_score(self):
        from tlbo.facade.obtl_es import ES
        surrogate = ES(self.config_space, self.source_hpo_data,
                       self.configuration_list, self.random_seed,
                       surrogate_type=self.surrogate_type,
                       num_src_hpo_trial=self.num_src_hpo_trial)
        X = convert_configurations_to_array(self.configuration_list)
        scores, _ = surrogate.predict(X)
        sorted_items = sorted(list(zip(self.configuration_list, scores)), key=lambda x: x[1])
        return [item[0] for item in sorted_items]

    def initial_design(self, n_init=3):
        initial_configs = list()
        configs_ = self.sort_configs_by_score()[:25]
        from sklearn.cluster import KMeans
        X = convert_configurations_to_array(configs_)
        kmeans = KMeans(n_clusters=n_init, random_state=0).fit(X)
        labels = kmeans.predict(X)
        label_set = set()
        for idx, _config in enumerate(configs_):
            if labels[idx] not in label_set:
                label_set.add(labels[idx])
                initial_configs.append(_config)
        return initial_configs

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            X = convert_configurations_to_array(self.configurations)
        Y = np.array(self.perfs, dtype=np.float64)
        # start_time = time.time()
        config = self.choose_next(X, Y)
        # print('In %d-th iter, config selection took %.3fs' % (self.iteration_id, time.time() - start_time))

        trial_state = SUCCESS
        trial_info = None

        if config not in (self.configurations + self.failed_configurations):
            # Evaluate this configuration.
            perf = self.target_hpo_measurements[config]
            if perf == MAXINT:
                trial_info = 'failed configuration evaluation.'
                trial_state = FAILDED
                self.logger.error(trial_info)

            if trial_state == SUCCESS and perf < MAXINT:
                if len(self.configurations) == 0:
                    self.default_obj_value = perf

                self.configurations.append(config)
                self.perfs.append(perf)
                self.history_container.add(config, perf)
            else:
                self.failed_configurations.append(config)
        else:
            self.logger.debug('This configuration has been evaluated! Skip it.')
            if config in self.configurations:
                config_idx = self.configurations.index(config)
                trial_state, perf = SUCCESS, self.perfs[config_idx]
            else:
                trial_state, perf = FAILDED, MAXINT

        self.iteration_id += 1
        self.logger.info(
            'Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config, trial_state, perf, trial_info

    def sample_random_config(self, config_num=1):
        configs = list()
        sample_cnt = 0
        while len(configs) < config_num:
            sample_cnt += 1
            _idx = self.rng.randint(len(self.configuration_list))
            config = self.configuration_list[_idx]
            if config not in (self.configurations + self.failed_configurations + configs):
                configs.append(config)
                sample_cnt = 0
            else:
                sample_cnt += 1
            if sample_cnt >= 200:
                configs.append(config)
                sample_cnt = 0
        return configs

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            if self.initial_configurations is None:
                default_config = self.config_space.get_default_configuration()
                if default_config not in (self.configurations + self.failed_configurations):
                    config = default_config
                else:
                    config = self.sample_random_config()[0]
                return config
            else:
                print('This is a config for warm-start!')
                return self.initial_configurations[_config_num]

        if self.random_configuration_chooser.check(self.iteration_id):
            config = self.sample_random_config()[0]
            if len(self.model.target_weight) == 0:
                self.model.target_weight.append(0.)
            else:
                self.model.target_weight.append(self.model.target_weight[-1])
            return config
        else:
            start_time = time.time()
            self.model.train(X, Y)
            print('Training surrogate model took %.3f' % (time.time() - start_time))

            incumbent_value = self.history_container.get_incumbents()[0][1]
            # if self.model.method_id == 'rgpe':
            #     y_, _, _ = zero_mean_unit_var_normalization(Y)
            #     incumbent_value = np.min(y_)

            if self.acq_func == 'ei':
                self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                                 num_data=len(self.history_container.data))
            elif self.acq_func == 'taf':
                self.acquisition_function.update_target_model(self.model.target_surrogate,
                                                              incumbent_value,
                                                              num_data=len(self.history_container.data),
                                                              model_weights=self.model.w)
            else:
                raise ValueError('invalid acquisition function ~ %s.' % self.acq_func)

            start_time = time.time()
            sorted_configs = self.acq_optimizer.maximize(
                runhistory=self.history_container,
                num_points=1
            )
            print('optimizing acq func took', time.time() - start_time)
            for _config in sorted_configs:
                if _config not in (self.configurations + self.failed_configurations):
                    return _config
            raise ValueError('The configuration in the SET (%d) is over' % len(self.configuration_list))

    def load_topk_configs(self, src_meta_features, tar_meta_feature, k=5, trial_num=50):
        dataset_array = self.scale_fit_meta_features(src_meta_features)
        tar_dataset_array = self.scale_transform_meta_features(tar_meta_feature)
        dis_list = list()
        for src_feature in dataset_array:
            dis_list.append(np.sqrt(np.sum(np.square(src_feature - tar_dataset_array))))
        sorted_dis_idx = np.argsort(dis_list)
        topk_id = sorted_dis_idx[:k]

        topk_configs = list()
        for id in topk_id:
            configs_id = list(self.source_hpo_data[id].items())[:trial_num]
            sorted_configs = sorted(configs_id, key=lambda x: x[1])
            topk_configs.append(sorted_configs[0][0])
        return topk_configs

    def scale_fit_meta_features(self, meta_features):
        from sklearn.preprocessing import MinMaxScaler, Imputer
        meta_features = np.array(meta_features)
        self.meta_feature_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.meta_feature_imputer.fit(meta_features)
        meta_features = self.meta_feature_imputer.transform(meta_features)
        self.meta_feature_scaler = MinMaxScaler()
        self.meta_feature_scaler.fit(meta_features)
        return self.meta_feature_scaler.transform(meta_features)

    def scale_transform_meta_features(self, meta_feature):
        _meta_features = np.array([meta_feature])
        _meta_feature = self.meta_feature_imputer.transform(_meta_features)
        _meta_feature = self.meta_feature_scaler.transform(_meta_feature)
        return np.clip(_meta_feature, 0, 1)

import time
import numpy as np
from typing import List, Dict
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter

from tlbo.model.util_funcs import get_rng, get_types
from tlbo.acquisition_function.acquisition import EI
from tlbo.config_space import ConfigurationSpace, Configuration
from tlbo.facade.notl import NoTL
from tlbo.optimizer.ei_offline_optimizer import OfflineSearch
from tlbo.optimizer.random_configuration_chooser import ChooserProb
from tlbo.config_space.util import convert_configurations_to_array
from tlbo.utils.constants import MAXINT, SUCCESS, FAILDED
from tlbo.utils.normalization import zero_mean_unit_var_normalization, zero_one_normalization
from tlbo.acquisition_function.ta_acquisition import TAQ_EI
from tlbo.framework.smbo import BasePipeline
from tlbo.facade.base_facade import BaseFacade


class SMBO_SEARCH_SPACE_Enlarge(BasePipeline):
    def __init__(self,
                 target_hpo_data: Dict,
                 config_space: ConfigurationSpace,
                 surrogate_model: BaseFacade,
                 acq_func: str = 'ei',
                 mode='best',
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
        self.mode = mode

        self.max_iterations = max_runs
        self.iteration_id = 0
        self.default_obj_value = MAXINT

        self.target_hpo_measurements = target_hpo_data
        self.configuration_list = list(self.target_hpo_measurements.keys())
        print('Target problem space: %d configurations' % len(self.configuration_list))
        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()
        self.reduce_cnt = 0

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
        self.acquisition_function = EI(self.model)
        self.space_classifier = None
        self.random_configuration_chooser = ChooserProb(
            prob=0.1,
            rng=np.random.RandomState(self.random_seed)
        )
        # self.prop_eta = 0.9
        # self.prop_init = 0.3

        # Set the parameter in metric.
        self.ys = list(self.target_hpo_measurements.values())
        self.y_max, self.y_min = np.max(self.ys), np.min(self.ys)

        self.reduce_cnt = 0
        self.p_min = 10
        self.p_max = 60
        self.use_correct_rate = False

        if self.mode in ['box', 'ellipsoid']:
            continuous_types = (UniformFloatHyperparameter, UniformIntegerHyperparameter)
            self.continuous_mask = [isinstance(hp, continuous_types) for hp in self.config_space.get_hyperparameters()]
            self.calculate_box_area()

    def get_adtm(self):
        y_inc = self.get_inc_y()
        assert self.y_max != self.y_min
        return (y_inc - self.y_min) / (self.y_max - self.y_min)

    def get_inc_y(self):
        # if isinstance(self.model, NoTL):
        #     _perfs = [_perf for (_, _perf) in list(self.target_hpo_measurements.items())[:self.iteration_id]]
        #     y_inc = np.min(_perfs)
        # else:
        #     y_inc = np.min(self.perfs)
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

    def evaluate(self, config):
        perf = self.target_hpo_measurements[config]
        return perf

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
            perf = self.evaluate(config)
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

    def sample_random_config(self, config_set=None, config_num=1):
        configs = list()
        sample_cnt = 0
        configurations = self.configuration_list if config_set is None else config_set
        while len(configs) < config_num:
            sample_cnt += 1
            _idx = self.rng.randint(len(configurations))
            config = configurations[_idx]
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
        """
        Step 1. sample a batch of random configs.
        Step 2. identify and preserve the configs in the good regions (union)
        Step 3. calculate their acquisition functions and choose the config with the largest value.
        Parameters
        ----------
        X
        Y

        Returns
        -------
        the config to evaluate next.
        """

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

        start_time = time.time()
        self.model.train(X, Y)
        print('Training surrogate model took %.3f' % (time.time() - start_time))

        if self.model.method_id in ['tst', 'tstm', 'pogpe']:
            y_, _, _ = zero_one_normalization(Y)
        elif self.model.method_id in ['scot']:
            y_ = Y.copy()
        else:
            y_, _, _ = zero_mean_unit_var_normalization(Y)
        incumbent_value = np.min(y_)

        if self.acq_func == 'ei':
            self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                             num_data=len(self.history_container.data))
        else:
            raise ValueError('invalid acquisition function ~ %s.' % self.acq_func)

        # Select space
        X_candidate = self.get_X_candidate()

        # Check space
        self.check_space(X_candidate)

        if self.rng.rand() < self.get_random_prob(self.iteration_id):
            excluded_set = list()
            candidate_set = set(X_candidate)
            for _config in self.configuration_list:
                if _config not in candidate_set:
                    excluded_set.append(_config)
            if len(excluded_set) == 0:
                excluded_set = self.configuration_list

            config = self.sample_random_config(config_set=excluded_set)[0]
            if len(self.model.target_weight) == 0:
                self.model.target_weight.append(0.)
            else:
                self.model.target_weight.append(self.model.target_weight[-1])
            print('Config sampled randomly.')
            return config

        acq_optimizer = OfflineSearch(X_candidate,
                                      self.acquisition_function,
                                      self.config_space,
                                      rng=np.random.RandomState(self.random_seed)
                                      )

        start_time = time.time()
        sorted_configs = acq_optimizer.maximize(
            runhistory=self.history_container,
            num_points=5000
        )
        print('Optimizing Acq. func took %.3f' % (time.time() - start_time))
        for _config in sorted_configs:
            if _config not in (self.configurations + self.failed_configurations):
                return _config

        print('[Warning] Reach unexpected?')
        excluded_set = list()
        candidate_set = set(X_candidate)
        for _config in self.configuration_list:
            if _config not in candidate_set and _config not in (self.configurations + self.failed_configurations):
                excluded_set.append(_config)
        if len(excluded_set) == 0:
            excluded_set = self.configuration_list
        return self.sample_random_config(config_set=excluded_set)[0]

    def get_X_candidate(self) -> List[Configuration]:
        if self.mode in ['box', 'ellipsoid']:
            return self.get_X_candidate_box()

        # Do task selection.
        if self.use_correct_rate:
            weights = self.model.correct_rate[:-1]  # exclude target weight
            print('use correct rate:', weights)
        else:
            weights = self.model.w[:-1]  # exclude target weight

        if self.mode == 'best':
            task_indexes = np.argsort(weights)[-1:]  # space
            task_indexes = [idx_ for idx_ in task_indexes if weights[idx_] > 0.]
        elif self.mode == 'all':
            task_indexes = np.argsort(weights)  # space-all
            task_indexes = [idx_ for idx_ in task_indexes if weights[idx_] > 0.]
        elif self.mode == 'sample':
            weights_ = [x / sum(weights) for x in weights]  # space-sample
            task_indexes = np.random.choice(list(range(len(weights))), 1, p=weights_)

        # Calculate the percentiles.
        p_min = self.p_min
        p_max = self.p_max
        percentiles = [p_max] * len(self.source_hpo_data)
        for _task_id in task_indexes:
            if self.use_correct_rate:
                _p = p_min + (1 - 2 * max(weights[_task_id] - 0.5, 0)) * (p_max - p_min)
            else:
                _p = p_min + (1 - weights[_task_id]) * (p_max - p_min)
            percentiles[_task_id] = _p
        print('Task Indexes', task_indexes)
        print('Percentiles', percentiles)
        self.prepare_classifier(task_indexes, percentiles)

        self.update_configuration_list()  # for online benchmark

        X_ALL = convert_configurations_to_array(self.configuration_list)
        y_pred = list()
        for _task_id in task_indexes:
            y_pred.append(self.space_classifier[_task_id].predict(X_ALL))

        X_candidate = list()
        if len(y_pred) > 0:
            # Count the #intersection.
            pred_mat = np.array(y_pred)
            # print(pred_mat.shape)
            # print(np.sum(pred_mat))
            _cnt = 0
            config_indexes = list()
            for _col in range(pred_mat.shape[1]):
                if (pred_mat[:, _col] == 1).all():
                    _cnt += 1
                    config_indexes.append(_col)
            print('The intersection of candidate space is %d.' % _cnt)

            for _idx in config_indexes:
                X_candidate.append(self.configuration_list[_idx])
            print('The candidate space size is %d.' % len(X_candidate))

            if len(X_candidate) == 0:
                print('[Warning] Intersect=0, please check!')
                # Deal with the space with no candidates.
                if len(self.configurations) <= 20:
                    X_candidate = self.configuration_list
                else:
                    print('[Warning] len(y_pred)=0. choose_config_target_space, please check!')
                    X_candidate = self.choose_config_target_space()
        else:
            X_candidate = self.choose_config_target_space()
        assert len(X_candidate) > 0
        return X_candidate

    def prepare_classifier(self, task_ids, percentiles):
        # Train the binary classifier.
        print('Train binary classifiers.')
        start_time = time.time()
        self.space_classifier = [None] * len(self.source_hpo_data)
        normalize = 'standardize'

        for _task_id in task_ids:
            hpo_evaluation_data = self.source_hpo_data[_task_id]
            percentile_v = percentiles[_task_id]

            print('.', end='')
            _X, _y = list(), list()
            for _config, _config_perf in hpo_evaluation_data.items():
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)
            X = X[:self.num_src_hpo_trial]
            y = y[:self.num_src_hpo_trial]

            if normalize == 'standardize':
                if (y == y[0]).all():
                    y[0] += 1e-4
                y, _, _ = zero_mean_unit_var_normalization(y)
            elif normalize == 'scale':
                if (y == y[0]).all():
                    y[0] += 1e-4
                y, _, _ = zero_one_normalization(y)
                y = 2 * y - 1.
            else:
                raise ValueError('Invalid parameter in norm.')

            percentile = np.percentile(y, percentile_v)
            unique_ys = sorted(list(set(y)))
            if len(unique_ys) >= 2 and percentile <= unique_ys[0]:
                percentile = unique_ys[1]

            space_label = np.array(np.array(y) < percentile)
            if (np.array(y) == percentile).all():
                raise ValueError('Assertion violation: The same eval values!')
            if (space_label[0] == space_label).all():
                space_label = np.array(np.array(y) < np.mean(y))
                if (space_label[0] == space_label).all():
                    raise ValueError('Warning: Label treatment triggers!')
                else:
                    print('Warning: Label treatment triggers!')

            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            # print('Labels', space_label)
            # print('sum', np.sum(space_label))
            clf.fit(X, space_label)
            self.space_classifier[_task_id] = clf
        print('Building base classifier took %.3fs.' % (time.time() - start_time))

    def get_random_prob(self, iter_id):
        if self.mode in ['ellipsoid', 'box']:   # todo
            return 0

        if iter_id <= 25:
            return 0.1
        else:
            return 0.1

    def update_configuration_list(self):
        return

    def check_space(self, X_candidate):
        print('Global Optimum:' + str(min(self.target_hpo_measurements.values())))
        best_perf_in_candidate = 1
        best_config = None
        for config in X_candidate:
            perf = self.target_hpo_measurements[config]
            if perf < best_perf_in_candidate:
                best_perf_in_candidate = perf
                best_config = config
        print('Current Optimum:' + str(best_perf_in_candidate))
        print('Optimum in space:' + str(bool(min(self.target_hpo_measurements.values()) == best_perf_in_candidate)))
        print("Reduced: %.2f/%.2f, Rate: %.2f" % (
            len(X_candidate), len(self.target_hpo_measurements), len(X_candidate) / len(self.target_hpo_measurements)))
        if len(X_candidate) != len(self.target_hpo_measurements):
            self.reduce_cnt += 1
        print("Reduced space is applied for %d iterations!" % self.reduce_cnt)

    def choose_config_target_space(self):
        return self.configuration_list
        # X_ALL = convert_configurations_to_array(self.configuration_list)
        # X_ = convert_configurations_to_array(self.configurations)
        # y_ = np.array(self.perfs, dtype=np.float64)
        # percentile = np.percentile(y_, 30)
        # label_ = np.array(y_ < percentile)
        #
        # if (y_ == percentile).all():
        #     raise ValueError('assertion violation: the same eval values!')
        # if (label_[0] == label_).all():
        #     label_ = np.array(y_ < np.mean(y_))
        #     print('Label treatment triggers!')
        #
        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        # clf.fit(X_, label_)
        # pred_ = clf.predict(X_ALL)
        #
        # X_candidate = list()
        # for _i, _config in enumerate(self.configuration_list):
        #     if pred_[_i] == 1:
        #         X_candidate.append(_config)
        # if len(X_candidate) == 0:
        #     print('Warning: feasible set is empty!')
        #     X_candidate = self.configuration_list
        # return X_candidate

    def calculate_box_area(self):
        """
        [NIPS 2019] Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning
        """
        incumbent_src_configs = []
        for hpo_evaluation_data in self.source_hpo_data:
            configs = list(hpo_evaluation_data.keys())[:self.num_src_hpo_trial]
            perfs = list(hpo_evaluation_data.values())[:self.num_src_hpo_trial]
            idx = np.argmin(perfs)
            incumbent_src_configs.append(configs[idx])
        X_incumbents = convert_configurations_to_array(incumbent_src_configs)
        # exclude categorical params
        X_incumbents_ = X_incumbents[:, self.continuous_mask]

        if self.mode == 'ellipsoid':
            raise NotImplementedError
        elif self.mode == 'box':
            self.src_X_min_ = np.min(X_incumbents_, axis=0)
            self.src_X_max_ = np.max(X_incumbents_, axis=0)
        else:
            raise ValueError(self.mode)

    def get_X_candidate_box(self) -> List[Configuration]:
        """
        [NIPS 2019] Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning
        """
        X_ALL = convert_configurations_to_array(self.configuration_list)
        # exclude categorical params
        X_ALL_ = X_ALL[:, self.continuous_mask]

        if self.mode == 'ellipsoid':
            raise NotImplementedError
        elif self.mode == 'box':
            valid_mask = np.logical_and(self.src_X_min_ <= X_ALL_, X_ALL_ <= self.src_X_max_).all(axis=1)
            valid_idx = np.where(valid_mask)[0].tolist()
            if len(valid_idx) == 0:
                print('[Warning] no candidates in box area!')
                X_candidate = self.configuration_list
            else:
                X_candidate = [self.configuration_list[i] for i in valid_idx]

            assert len(X_candidate) > 0
            return X_candidate
        else:
            raise ValueError(self.mode)

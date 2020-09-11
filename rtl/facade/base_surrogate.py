import numpy as np
import george
from rtl.model.gp import GaussianProcess
from rtl.priors.default_priors import DefaultPrior


class BaseSurrogate(object):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='SE',
                 normalize_output=False, init=-1):
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.current_model = None
        self.historical_model = list()
        self.historical_task_num = len(train_metadata)
        self.n_dims = train_metadata[0].shape[1] - 1
        self.weights = [0.] * self.historical_task_num
        self.incumbent_x = None
        self.incumbent_y = None
        self.scaled_incumbent_y = None
        self.kernel_type = kernel_type
        self.normalize_output = normalize_output
        self.cov_amp = cov_amp
        self._init = init
        self._debug_mode = False
        self.scale = False

    def create_single_gp(self, lower, upper):
        initial_ls = np.ones([self.n_dims])
        if self.kernel_type == 'Matern':
            exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=self.n_dims)
        elif self.kernel_type == 'SE':
            exp_kernel = george.kernels.ExpSquaredKernel(initial_ls, ndim=self.n_dims)
        else:
            raise ValueError('Unsupported Kernel Type!')

        kernel = self.cov_amp * exp_kernel
        prior = DefaultPrior(len(kernel) + 1)
        model = GaussianProcess(kernel, prior=prior, normalize_output=self.normalize_output, normalize_input=False,
                                lower=lower, upper=upper)
        return model

    def update_incumbent(self, X, y):
        best_idx = np.argmax(y)
        self.incumbent_x, self.incumbent_y = X[best_idx], y[best_idx]

    def update_scaled_incumbent(self, y):
        best_idx = np.argmax(y)
        self.scaled_incumbent_y = y[best_idx]

    def get_incumbent(self, scaled=False):
        if self.scale and scaled:
            return self.incumbent_x, self.scaled_incumbent_y
        else:
            return self.incumbent_x, self.incumbent_y

    def prior_predict(self, x: np.array):
        assert len(self.historical_model) == self.historical_task_num
        X = x.reshape(1, len(x))
        # Target surrogate predictions with weight.
        mu, var = 0, 0

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.historical_task_num):
            weight = self.weights[i]
            mu_t, var_t = self.historical_model[i].predict(X)
            mu += weight * mu_t[0]
            var += weight * weight * var_t[0]
        return mu, var

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, init_type):
        self._init = init_type

    def set_mode(self, mode):
        self._debug_mode = mode

    # This method returns K configurations for warm starting.
    # init = 0: for the real-world HPO problem.
    # init = 1: designed for the controlled HPO problem in benchmark - ResNet50 and Xgboost.
    def choose_ws_configurations(self, K):
        assert self.init != -1
        config_list = list()
        if self.init == 0:
            best_list = self.choose_candidates()
            if len(best_list) <= K:
                return best_list
            # Sort the configs based on the WS strategy.
            performance_list = list()
            for i, item in enumerate(best_list):
                performance_list.append((self.prior_predict(item)[0], item))
            for perf, config in sorted(performance_list, key=lambda x: x[0], reverse=True)[:4*K]:
                config_list.append(config)
            config_list = self.choose_diverse_configurations(config_list, K)
        elif self.init == 1:
            assert len(self.test_metadata[:, 1:]) >= 2*K
            performance_list = list()
            for i, item in enumerate(self.test_metadata[:, 1:]):
                performance_list.append((self.prior_predict(item)[0], item))
            for perf, config in sorted(performance_list, key=lambda x: x[0], reverse=True)[:4*K]:
                config_list.append(config)
            config_list = self.choose_diverse_configurations(config_list, K)
        else:
            raise ValueError('Invalid init strategy')
        return config_list

    def choose_candidates(self):
        candidates = list()
        for i in range(self.historical_task_num):
            metadata = np.array(self.train_metadata[i])
            best_config = metadata[np.argmax(metadata[:, 0]), 1:]
            is_in = False
            for item in candidates:
                if (item == best_config).all():
                    is_in = True
                    break
            if not is_in:
                candidates.append(best_config)
        return candidates

    def choose_diverse_configurations(self, configs, K):
        assert len(configs) > K
        ws_candidates = [configs[0]]
        del configs[0]

        def diff_dist(conf1, conf2):
            diff_cnt = 0
            for val1, val2 in zip(conf1, conf2):
                diff_cnt += val1 != val2
            return diff_cnt

        def euclidean_dist(conf1, conf2):
            return np.linalg.norm(conf1 - conf2)

        for _ in range(K-1):
            # Compute the field-diff distance.
            dist_list = np.zeros(len(configs))
            for i, config in enumerate(configs):
                for ws_config in ws_candidates:
                    dist_list[i] += diff_dist(config, ws_config)
            indexs = np.argsort(-1*dist_list)
            pivot = dist_list[indexs[0]]
            index_candidates = [indexs[0]]
            for index in indexs[1:]:
                if dist_list[index] == pivot:
                    index_candidates.append(index)
                else:
                    break

            if len(indexs) == 1 or dist_list[indexs[0]] != dist_list[indexs[1]]:
                ws_candidates.append(configs[indexs[0]])
                del configs[indexs[0]]
            else:
                # Compute the euclidean distance.
                euclidean_list = np.zeros(len(index_candidates))
                for i, index in enumerate(index_candidates):
                    for ws_config in ws_candidates:
                        euclidean_list[i] += euclidean_dist(configs[index], ws_config)
                sec_indexs = np.argsort(-1 * euclidean_list)
                root_index = index_candidates[sec_indexs[0]]
                ws_candidates.append(configs[root_index])
                del configs[root_index]
        assert len(ws_candidates) == K
        return ws_candidates

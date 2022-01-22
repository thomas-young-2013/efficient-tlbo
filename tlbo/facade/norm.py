import numpy as np
from tlbo.facade.base_facade import BaseFacade


class NORM(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed, norm=3,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'norm'
        self.only_source = only_source
        self.build_source_surrogates(normalize='standardize')
        # Weights for space transfer
        self.w = [1. / self.K] * self.K + [0.]
        # Weights for surrogate
        self.sw = [1. / self.K] * self.K + [0.]
        self.scale = True

        # Preventing weight dilution.
        self.ignored_flag = [False] * self.K
        self.hist_ws = list()
        self.iteration_id = 0

        self.norm = norm

        self.correct_rate = None

        self.increasing_weight = False
        self.nondecreasing_weight = False
        self.same = True

    def train(self, X: np.ndarray, y: np.array):
        # Train the target surrogate and update the weight w.
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu)
            var_list.append(var)

        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='standardize')

        # Pretrain the leave-one-out surrogates.
        k_fold_num = 5
        cached_mu_list, cached_var_list = list(), list()
        instance_num = len(y)
        skip_target_surrogate = False if instance_num >= k_fold_num else True
        # Ignore the target surrogate.
        # skip_target_surrogate = True

        if not skip_target_surrogate:
            # Conduct leave-one-out evaluation.
            if instance_num < k_fold_num:
                for i in range(instance_num):
                    row_indexs = list(range(instance_num))
                    del row_indexs[i]
                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-4
                    model = self.build_single_surrogate(X[row_indexs, :], y[row_indexs], normalize='standardize')
                    mu, var = model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)
            else:
                # Conduct K-fold cross validation.
                fold_num = instance_num // k_fold_num
                for i in range(k_fold_num):
                    row_indexs = list(range(instance_num))
                    bound = (instance_num - i * fold_num) if i == (k_fold_num - 1) else fold_num
                    for index in range(bound):
                        del row_indexs[i * fold_num]

                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-4

                    model = self.build_single_surrogate(X[row_indexs, :], y[row_indexs], normalize='standardize')
                    mu, var = model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)

        ranking_loss_list = list()
        for id in range(self.K):
            sampled_y = mu_list[id].copy()
            rank_loss = 0
            for i in range(len(y)):
                for j in range(len(y)):
                    if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                        rank_loss += 1
            ranking_loss_list.append(rank_loss)

        # Compute ranking loss for target surrogate.
        rank_loss = 0
        if not skip_target_surrogate:
            if instance_num < k_fold_num:
                for i in range(instance_num):
                    sampled_y = cached_mu_list[i].copy()
                    for j in range(instance_num):
                        if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                            rank_loss += 1
            else:
                fold_num = instance_num // k_fold_num
                for fold in range(k_fold_num):
                    sampled_y = cached_mu_list[fold].copy()
                    bound = instance_num if fold == (k_fold_num - 1) else (fold + 1) * fold_num
                    for i in range(fold_num * fold, bound):
                        for j in range(instance_num):
                            if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                rank_loss += 1
        else:
            rank_loss = instance_num * instance_num
        ranking_loss_list.append(rank_loss)

        # Update the weights.
        correct_rate_list = [1 - x / (instance_num ** 2) for x in ranking_loss_list]
        self.correct_rate = np.array(correct_rate_list)
        norm_sum = sum([x ** self.norm for x in correct_rate_list])
        for id in range(self.K + 1):
            self.w[id] = pow(correct_rate_list[id], self.norm) / norm_sum
        print(correct_rate_list, norm_sum)

        # # Set weight dilution flag.
        # ranking_loss_caches = np.array(ranking_loss_caches)
        # threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        # for id in range(self.K):
        #     median = sorted(ranking_loss_caches[:, id])[int(self.num_sample * 0.5)]
        #     self.ignored_flag[id] = median > threshold

        if self.only_source:
            self.w[-1] = 0.
            if np.sum(self.w) == 0:
                self.w = [1. / self.K] * self.K + [0.]
            else:
                self.w[:-1] = np.array(self.w[:-1]) / np.sum(self.w[:-1])

        if self.nondecreasing_weight:
            old_weights = np.array(self.sw.copy())
            new_weights = np.array(self.w.copy())
            old_last_weight = old_weights[-1]
            new_last_weight = new_weights[-1]
            if new_last_weight < old_last_weight:
                old_remain_weight = 1.0 - old_last_weight
                new_remain_weight = 1.0 - new_last_weight
                if new_remain_weight <= 1e-8:
                    adjusted_new_weights = np.array([0.] * self.K + [1.], dtype=np.float64)
                else:
                    adjusted_new_weights = np.append(new_weights[:-1] / new_remain_weight * old_remain_weight,
                                                     old_last_weight)
                self.sw = adjusted_new_weights.copy()
            else:
                self.sw = new_weights.copy()
        elif self.increasing_weight and instance_num > 10:
            # Increasing target
            new_weights = np.array(self.w.copy())
            s = 10
            k = 0.04  # Speed
            a = 0.5
            new_last_weight = a / (a + np.e ** (-(instance_num - s) * k))
            new_remain_weight = 1.0 - new_last_weight
            remain_weight = 1.0 - new_weights[-1]
            if remain_weight <= 1e-8:
                adjusted_new_weights = np.array([0.] * self.K + [1.], dtype=np.float64)
            else:
                adjusted_new_weights = np.append(new_weights[:-1] / remain_weight * new_remain_weight,
                                                 new_last_weight)
            self.sw = adjusted_new_weights
        else:
            self.sw = self.w.copy()

        if self.same:
            self.w = self.sw.copy()

        print('=' * 20)
        w = self.w.copy()
        # for id in range(self.K):
        #     if self.ignored_flag[id]:
        #         w[id] = 0.

        space_weight_str = ','.join([('%.2f' % item) for item in w])
        surrogate_weight_str = ','.join([('%.2f' % item) for item in self.sw])
        print('In iter-%d' % self.iteration_id)
        self.target_weight.append(w[-1])
        print('Space weight:' + space_weight_str)
        print('Surrogate weight:' + surrogate_weight_str)
        self.hist_ws.append(w)
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        # Target surrogate predictions with weight.
        mu *= self.sw[-1]
        var *= (self.sw[-1] * self.sw[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.K):
            if not self.ignored_flag[i]:
                mu_t, var_t = self.source_surrogates[i].predict(X)
                mu += self.sw[i] * mu_t
                var += self.sw[i] * self.sw[i] * var_t
        return mu, var

    def get_weights(self):
        return self.w

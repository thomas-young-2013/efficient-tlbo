import numpy as np
from tlbo.facade.base_facade import BaseFacade


class TSTSPACE(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, use_metafeatures=False, metafeatures=None,
                 only_source=False):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'tst_space'
        self.only_source = only_source
        self.build_source_surrogates(normalize='scale')
        # Weights for space transfer
        self.w = [1. / self.K] * self.K + [0.]
        # Weights for base surrogates and the target surrogate.
        self.sw = [0.75] * (self.K + 1)
        self.use_metafeatures = use_metafeatures
        self.metafeatures = metafeatures
        self.bandwidth = 0.45
        if self.use_metafeatures:
            assert len(metafeatures) == self.K + 1
            self.meta_dist = [0.] * self.K
            for _id in range(self.K):
                _dist = np.power(self.metafeatures[self.K] - self.metafeatures[_id], 2)
                self.meta_dist = _dist

        # Preventing weight dilution.
        self.hist_ws = list()
        self.iteration_id = 0

        self.norm = 3

    def train(self, X: np.ndarray, y: np.array):
        # Train the target surrogate and update the weight w.
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu)
            var_list.append(var)

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

        # Compute weights for space transfer.
        ranking_loss_list = list()
        for id in range(self.K):
            sampled_y = mu_list[id].copy()
            rank_loss = 0
            for i in range(len(y)):
                for j in range(len(y)):
                    if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                        rank_loss += 1
            ranking_loss_list.append(rank_loss)

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

        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize='scale')

        n_sample = X.shape[0]
        for _id in range(self.K):
            if not self.use_metafeatures:
                mu, _ = self.source_surrogates[_id].predict(X)
                discordant_paris, total_pairs = 0, 0
                for i in range(n_sample):
                    for j in range(i + 1, n_sample):
                        if (y[i] < y[j]) ^ (mu[i] < mu[j]):
                            discordant_paris += 1
                        total_pairs += 1
                tmp = discordant_paris / total_pairs / self.bandwidth
            else:
                tmp = self.meta_dist[_id] / self.bandwidth
            print(tmp)
            self.sw[_id] = 0.75 * (1 - tmp * tmp) if tmp <= 1 else 0

        if self.only_source:
            self.sw[-1] = 0.
            self.sw = np.array(self.sw) / np.sum(self.sw)

        print('=' * 20)
        w = self.sw.copy()
        space_weight_str = ','.join([('%.2f' % item) for item in self.w])
        surrogate_weight_str = ','.join([('%.2f' % item) for item in self.sw])
        print('In iter-%d' % self.iteration_id)
        self.target_weight.append(w[-1])
        print('Space weight:' + space_weight_str)
        print('Surrogate weight:' + surrogate_weight_str)

        self.target_weight.append(w[-1] / sum(w))
        self.hist_ws.append(w)
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        denominator = 0.75
        for i in range(0, self.K):
            weight = self.sw[i]
            mu_t, _ = self.source_surrogates[i].predict(X)
            mu += weight * mu_t
            denominator += weight
        mu /= denominator
        return mu, var

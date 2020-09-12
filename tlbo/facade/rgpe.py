import numpy as np
from tlbo.facade.base_facade import BaseFacade


class RGPE(BaseFacade):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, rng, surrogate_type='gp'):
        super().__init__(config_space, source_hpo_data, rng, target_hp_configs, surrogate_type=surrogate_type)
        
        self.build_source_surrogates()
        # Weights for base surrogates and the target surrogate.
        self.w = [1./self.K]*self.K + [0.]
        self.scale = True
        self.num_sample = np.max([50, (self.K + 1) * 5])

        # Preventing weight dilution.
        self.ignored_flag = [False] * self.K
    
    def train(self, X: np.ndarray, y: np.array):
        # Train the target surrogate and update the weight w.
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu)
            var_list.append(var)

        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y)

        # Pretrain the leave-one-out surrogates.
        k_fold_num = 5
        cached_mu_list, cached_var_list = list(), list()
        instance_num = len(y)
        skip_target_surrogate = False if instance_num >= k_fold_num else True

        if not skip_target_surrogate:
            # Conduct leave-one-out evaluation.
            self.build_single_surrogate(X, y)
            if instance_num < k_fold_num:
                for i in range(instance_num):
                    row_indexs = list(range(instance_num))
                    del row_indexs[i]
                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-4
                    model = self.build_single_surrogate(X[row_indexs, :], y[row_indexs])
                    mu, var = model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)
            else:
                # Conduct K-fold cross validation.
                fold_num = instance_num // k_fold_num
                for i in range(k_fold_num):
                    row_indexs = list(range(instance_num))
                    bound = (instance_num - i*fold_num) if i == (k_fold_num-1) else fold_num
                    for index in range(bound):
                        del row_indexs[i*fold_num]

                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-4

                    model = self.build_single_surrogate(X[row_indexs, :], y[row_indexs])
                    mu, var = model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)

        argmin_list = [0] * (self.K + 1)
        ranking_loss_caches = list()
        for _ in range(self.num_sample):
            ranking_loss_list = list()
            for id in range(self.K):
                sampled_y = np.random.normal(mu_list[id], var_list[id])
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
                        sampled_y = np.random.normal(cached_mu_list[i], cached_var_list[i])
                        for j in range(instance_num):
                            if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                rank_loss += 1
                else:
                    fold_num = instance_num // k_fold_num
                    for fold in range(k_fold_num):
                        sampled_y = np.random.normal(cached_mu_list[fold], cached_var_list[fold])
                        bound = instance_num if fold == (k_fold_num-1) else (fold+1)*fold_num
                        for i in range(fold_num*fold, bound):
                            for j in range(instance_num):
                                if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                    rank_loss += 1
            else:
                rank_loss = instance_num * instance_num
            ranking_loss_list.append(rank_loss)
            ranking_loss_caches.append(ranking_loss_list)

            argmin_task = np.argmin(ranking_loss_list)
            argmin_list[argmin_task] += 1

        # Update the weights.
        for id in range(self.K + 1):
            self.w[id] = argmin_list[id] / self.num_sample

        # Set weight dilution flag.
        ranking_loss_caches = np.array(ranking_loss_caches)
        threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        for id in range(self.K):
            median = sorted(ranking_loss_caches[:, id])[int(self.num_sample * 0.5)]
            self.ignored_flag[id] = median > threshold

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        # Target surrogate predictions with weight.
        mu *= self.w[-1]
        var *= (self.w[-1] * self.w[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.K):
            if not self.ignored_flag[i]:
                mu_t, var_t = self.source_surrogates[i].predict(X)
                mu += self.w[i] * mu_t
                var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_weights(self):
        return self.w

import numpy as np

from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class RankingWeightedEnsembleSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        # Weights vector for all surrogates, including the target model.
        self.weights = [1./self.historical_task_num]*self.historical_task_num + [0.]
        self.scale = True
        self.num_sample = (self.historical_task_num + 1) * 5

        # Preventing weight dilution.
        self.ignored_flag = [False] * self.historical_task_num

        # Train each individual surrogate.
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            model = self.create_single_gp(lower, upper)
            # Prevent the same value in y.
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
            model.train(X, y)
            self.historical_model.append(model)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Train the current task's surrogate and update the weights.
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)

        mu_list = list()
        var_list = list()
        for task_id in range(self.historical_task_num):
            mu, var = self.historical_model[task_id].predict(X)
            mu_list.append(mu)
            var_list.append(var)

        # Pretrain the leave-one-out surrogates.
        cached_mu_list = list()
        cached_var_list = list()

        instance_num = len(y)
        skip_target_surrogate = False if instance_num >= 3 else True
        if skip_target_surrogate:
            self.update_scaled_incumbent([0]*len(y))
        else:
            # Standardize the y with normal distribution.
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
            self.update_scaled_incumbent(y)

        if not skip_target_surrogate:
            # Conduct leave-one-out evaluation.
            self.current_model = self.create_single_gp(lower, upper)
            if instance_num < 10:
                for i in range(instance_num):
                    row_indexs = list(range(instance_num))
                    del row_indexs[i]
                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-5
                    do_optimize = True if i == 0 else False
                    self.current_model.train(X[row_indexs, :], y[row_indexs], do_optimize=do_optimize)
                    mu, var = self.current_model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)
            else:
                # Conduct 5-fold evaluation.
                K = 3
                fold_num = instance_num // K
                for i in range(K):
                    row_indexs = list(range(instance_num))
                    bound = (instance_num - i*fold_num) if i == (K-1) else fold_num
                    for index in range(bound):
                        del row_indexs[i*fold_num]

                    if (y[row_indexs] == y[row_indexs[0]]).all():
                        y[row_indexs[0]] += 1e-5
                    do_optimize = True if i == 0 else False
                    self.current_model.train(X[row_indexs, :], y[row_indexs], do_optimize=do_optimize)
                    mu, var = self.current_model.predict(X)
                    cached_mu_list.append(mu)
                    cached_var_list.append(var)

        argmin_list = [0] * (self.historical_task_num + 1)
        ranking_loss_caches = list()
        for _ in range(self.num_sample):
            ranking_loss_list = []
            for task_id in range(self.historical_task_num):
                sampled_y = np.random.normal(mu_list[task_id], var_list[task_id])

                rank_loss = 0
                for i in range(len(y)):
                    for j in range(len(y)):
                        if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                            rank_loss += 1
                ranking_loss_list.append(rank_loss)

            # Compute ranking loss for target surrogate.
            rank_loss = 0
            if not skip_target_surrogate:
                if instance_num < 10:
                    for i in range(instance_num):
                        sampled_y = np.random.normal(cached_mu_list[i], cached_var_list[i])
                        for j in range(instance_num):
                            if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                rank_loss += 1
                else:
                    fold_num = instance_num // 5
                    for fold in range(K):
                        sampled_y = np.random.normal(cached_mu_list[fold], cached_var_list[fold])
                        bound = instance_num if fold == (K-1) else (fold+1)*fold_num
                        for i in range(fold_num*fold, bound):
                            for j in range(instance_num):
                                if (y[i] < y[j]) ^ (sampled_y[i] < sampled_y[j]):
                                    rank_loss += 1
            else:
                rank_loss = len(y) * len(y)
            ranking_loss_list.append(rank_loss)
            ranking_loss_caches.append(ranking_loss_list)

            argmin_task = np.argmin(ranking_loss_list)
            argmin_list[argmin_task] += 1

        # Update the weights.
        for task_id in range(self.historical_task_num + 1):
            self.weights[task_id] = argmin_list[task_id] / self.num_sample

        # Set weight dilution flag.
        ranking_loss_caches = np.array(ranking_loss_caches)
        threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        for task_id in range(self.historical_task_num):
            median = sorted(ranking_loss_caches[:, task_id])[int(self.num_sample * 0.5)]
            self.ignored_flag[task_id] = median > threshold

    def predict(self, X: np.array):
        # Predict the given x's objective value (mean, std).
        # The predicting result is influenced by the ensemble surrogate with weights.
        n = X.shape[0]
        # If there is no metadata for current model, output the determinant prediction: 0., 0.
        if self.current_model is None:
            mu, var = np.zeros(n), np.zeros(n)
        else:
            mu, var = self.current_model.predict(X)
        # Target surrogate predictions with weight.
        mu *= self.weights[-1]
        var *= (self.weights[-1] * self.weights[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.historical_task_num):
            if not self.ignored_flag[i]:
                weight = self.weights[i]
                mu_t, var_t = self.historical_model[i].predict(X)
                mu += weight * mu_t
                var += weight * weight * var_t
        return mu, var

    def get_weights(self):
        return self.weights

import math
import numpy as np

from tlbo.facade.base_surrogate import BaseSurrogate


@DeprecationWarning
class RobustEnsembleSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, model_type=1, strategy=0, cov_amp=2, kernel_type='Matern'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        # Weights vector for all surrogates, including the target model.
        self.weights = [1./(self.historical_task_num + 1)]*(self.historical_task_num + 1)
        self.bandwidth = 0.5
        self.ratio = 0.5
        self.model_type = model_type
        self.strategy = strategy
        # Train each individual surrogate.
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            model = self.create_single_gp(lower, upper)
            model.train(X, y)
            self.historical_model.append(model)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Train the current task's surrogate and update the weights.
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)

        self.current_model = self.create_single_gp(lower, upper)
        self.current_model.train(X, y)

        # Evaluate the target surrogate.
        mu, _ = self.current_model.predict(X)
        discordant_paris, total_pairs = 0, 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if (y[i] < y[j]) ^ (mu[i] < mu[j]):
                    discordant_paris += 1
                total_pairs += 1
        tmp = discordant_paris / total_pairs / self.bandwidth
        self.weights[self.historical_task_num] = 0.75 * (1 - tmp * tmp) if tmp <= 1 else 0

        # TODO: standardize the y with normal distribution.

        for task_id in range(self.historical_task_num):
            mu, _ = self.historical_model[task_id].predict(X)
            discordant_paris, total_pairs = 0, 0
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    if (y[i] < y[j]) ^ (mu[i] < mu[j]):
                        discordant_paris += 1
                    total_pairs += 1
            tmp = discordant_paris / total_pairs / self.bandwidth
            self.weights[task_id] = 0.75*(1-tmp*tmp) if tmp <= 1 else 0

    def get_dynamic_weight(self, X):
        weights = np.array(self.weights.copy())
        bandwidth = 0.0
        mu_list, var_list = list(), list()
        for i in range(0, self.historical_task_num):
            mu, var = self.historical_model[i].predict(X)
            mu_list.append(mu[0])
            var_list.append(var[0])

        mu, var = self.current_model.predict(X)
        mu_list.append(mu[0])
        var_list.append(var[0])
        if self.strategy == 0:
            vars = np.array(var_list)
            vars = vars/np.mean(vars)
            deltas = np.array(list(map(lambda x: 1-x*x if x <= 1 else 1/(x*x)-1, vars)))
            weights = np.add(weights, bandwidth * deltas)
            weights = list(map(lambda x: 0 if x <= 0 else x, weights))
        elif self.strategy == 1:
            pivot = sorted(var_list)[int(len(var_list) * self.ratio)]
            weights = list(map(lambda x: 0 if x > pivot else x, weights))
        return weights

    def predict(self, X: np.array, dynamic=False):
        # Predict the given x's objective value (mean, std).
        # The predicting result is influenced by the ensemble surrogate with weights.
        weights = np.array(self.weights.copy())
        if dynamic:
            weights = self.get_dynamic_weight(X)

        # If there is no metadata for current model, output the determinant prediction: 0., 0.
        mu_list, var_list = list(), list()
        for i in range(0, self.historical_task_num):
            mu_t, var_t = self.historical_model[i].predict(X)
            mu_list.append(mu_t[0])
            var_list.append(var_t[0])
        mu_t, var_t = self.current_model.predict(X)
        mu_list.append(mu_t[0])
        var_list.append(var_t[0])

        mu, var = 0, 0
        if self.model_type == 4:
            weights = weights/sum(weights)
            for i in range(0, self.historical_task_num + 1):
                mu += weights[i] * mu_list[i]
            for i in range(0, self.historical_task_num + 1):
                var += weights[i] * (mu_list[i]*mu_list[i] + var_list[i])
            var -= mu*mu
            if var < 0.:
                print('var less than 0.')
                var = 1e-4
            return mu, var

        cov_mat = None
        if self.model_type == 3:
            cov_mat = 0.
            n_sample = 100
            observations = list()
            for i in range(n_sample):
                sampled_y = np.random.normal(mu_list, var_list)
                observations.append(sampled_y)
            observations = np.array(observations)
            N = self.historical_task_num + 1
            ob_mean = np.mean(observations, axis=0)
            for i in range(N):
                cov_mat += (observations[i] - ob_mean).reshape((N, 1)) * (observations[i] - ob_mean).reshape((1, N))
            cov_mat = cov_mat/(N-1.)
            assert cov_mat.shape == (N, N)

        # Target surrogate predictions with weight.
        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.historical_task_num + 1):
            weight = weights[i]
            mu_t, var_t = mu_list[i], var_list[i]
            mu += weight * mu_t
            var += weight * weight * var_t
        if self.model_type == 2 or self.model_type == 3:
            for i in range(self.historical_task_num + 1):
                for j in range(i + 1, self.historical_task_num + 1):
                    if self.model_type == 2:
                        cov_estimate = math.sqrt(var_list[i] * var_list[j])
                    elif self.model_type == 3:
                        cov_estimate = cov_mat[i][j]
                    else:
                        raise ValueError('Invalid model type!', self.model_type)
                    var += 2 * self.weights[i] * self.weights[j] * cov_estimate
        return mu, var

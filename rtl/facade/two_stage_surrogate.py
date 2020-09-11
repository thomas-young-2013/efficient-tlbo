import numpy as np
from rtl.facade.base_surrogate import BaseSurrogate
from rtl.utils.normalization import zero_one_normalization


class TwoStageSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern',
                 use_ranking=True, bandwidth=0.1, hp_range=6):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.bandwidth = bandwidth
        self.weights = [0.75] * self.historical_task_num
        self.use_ranking = use_ranking
        self.hp_range = hp_range
        self.meta_feature_dist = [0.] * self.historical_task_num
        test_meta_feature = test_metadata[0, hp_range+1:]
        # Train each individual surrogate.
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            if (y == y[0]).all():
                y[0] += 1e-4
            if not self.use_ranking:
                meta_feature = X[0, hp_range:]
                self.meta_feature_dist[i] = np.linalg.norm(test_meta_feature-meta_feature, 2)
                X = X[:, :hp_range]
                self.n_dims = hp_range

            # Scale the instance in training meta-dataset to [0, 1].
            y, _, _ = zero_one_normalization(y)
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)
            model = self.create_single_gp(lower, upper)
            model.train(X, y)
            self.historical_model.append(model)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Train the current task's surrogate and update the weights.
        if not self.use_ranking:
            X = X[:, :self.hp_range]
        lower = np.amin(X, axis=0)
        upper = np.amax(X, axis=0)

        self.current_model = self.create_single_gp(lower, upper)
        # The scale for the new dataset remains untouched.
        self.current_model.train(X, y)

        for task_id in range(self.historical_task_num):
            if len(X) < 2:
                self.weights[task_id] = 0.75
                continue
            if self.use_ranking:
                mu, _ = self.historical_model[task_id].predict(X)
                discordant_paris, total_pairs = 0, 0
                for i in range(len(X)):
                    for j in range(i+1, len(X)):
                        if (y[i] < y[j]) ^ (mu[i] < mu[j]):
                            discordant_paris += 1
                        total_pairs += 1
                tmp = discordant_paris / total_pairs / self.bandwidth
            else:
                tmp = self.meta_feature_dist[task_id] / self.bandwidth
            self.weights[task_id] = 0.75*(1-tmp*tmp) if tmp <= 1 else 0

    def predict(self, X: np.array):
        # Predict the given x's objective value (mean, std).
        # The predicting result is influenced by the ensemble surrogate with weights.
        # If there is no metadata for current model, then output the determinant prediction: 0, 1000.
        if not self.use_ranking:
            X = X[:, :self.hp_range]
        mu, var = self.current_model.predict(X)
        denominator = 0.75
        for i in range(0, self.historical_task_num):
            weight = self.weights[i]
            mu_t, _ = self.historical_model[i].predict(X)
            mu += weight * mu_t
            denominator += weight
        mu /= denominator
        return mu, var

    def get_weights(self):
        return self.weights

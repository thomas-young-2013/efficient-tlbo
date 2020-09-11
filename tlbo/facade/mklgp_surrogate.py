import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.model.mkl_gp import MKLGaussianProcess
from tlbo.utils.normalization import zero_mean_unit_var_normalization


class MKLGPSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern', base_model='gp'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.scale = True
        self.base_model_type = base_model
        self.split = 7
        metafeatures = list()
        self.X, self.y = None, None
        for X in train_metadata:
            metafeatures.append(X[0][self.split:])
            y = X[:, 0]
            if (y == y[0]).all():
                y[0] += 1e-5
            y, _, _ = zero_mean_unit_var_normalization(y)
            X = X[:, 1:]
            if self.X is not None:
                self.X = np.r_[self.X, X]
                self.y = np.r_[self.y, y]
            else:
                self.X, self.y = X, y
        metafeatures.append(test_metadata[0][self.split:])
        self.current_model = MKLGaussianProcess(metafeatures)
        self.iter = 1

    def train(self, X: np.ndarray, y: np.array):
        print('iter', self.iter)
        self.iter += 1
        train_size = y.shape[0]
        self.update_incumbent(X, y)
        X = np.r_[self.X, X]
        if len(y) > 2:
            if (y == y[0]).all():
                y[0] += 1e-5
            y, _, _ = zero_mean_unit_var_normalization(y)
            y = np.r_[self.y, y]
        else:
            y = self.y
        self.update_scaled_incumbent(y[-train_size:])
        # use_optimize = True if self.iter%3 == 0 else False
        self.current_model.train(X, y, optimize=False)
        print('iter finishes', self.iter)

    def predict(self, X: np.array):
        mu, var = self.current_model.predict(X)
        return mu, var

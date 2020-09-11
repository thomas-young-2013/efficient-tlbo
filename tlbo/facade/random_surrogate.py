import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate


class RandomSearch(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata):
        BaseSurrogate.__init__(self, train_metadata, test_metadata)

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

    def predict(self, X: np.array):
        # Imitate the random search.
        n = X.shape[0]
        mu = np.random.rand(n)
        var = np.array([1e-4]*n)
        return mu, var

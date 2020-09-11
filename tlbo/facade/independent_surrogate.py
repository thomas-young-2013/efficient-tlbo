import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate
from tlbo.model.rf_with_instances import RandomForestWithInstances


class IndependentSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern', base_model='gp'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.base_model_type = base_model

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        if self.base_model_type == 'gp':
            # Train the current task's surrogate and update the weights.
            lower = np.amin(X, axis=0)
            upper = np.amax(X, axis=0)

            self.current_model = self.create_single_gp(lower, upper)
        else:
            self.current_model = RandomForestWithInstances()
        self.current_model.train(X, y)

    def predict(self, X: np.array):
        # If there is no metadata for current model, output the determinant prediction: 0., 0.
        mu, var = self.current_model.predict(X)
        return mu, var

import numpy as np
from rtl.facade.base_surrogate import BaseSurrogate
from rtl.utils.rank_svm import RankSVM


class SCoTSurrogate(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='SE'):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.X, self.y = None, None
        self.scale = True
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            num_sample = X.shape[0]
            y = np.c_[self.train_metadata[i][:, 0], np.array([i]*num_sample)]
            if self.X is not None:
                self.X = np.r_[self.X, X]
                self.y = np.r_[self.y, y]
            else:
                self.X, self.y = X, y
        rank_svm = RankSVM()
        rank_svm.fit(self.X, self.y)
        self.pred_y = rank_svm.predict(self.X)

        lower = np.amin(self.X, axis=0)
        upper = np.amax(self.X, axis=0)
        self.current_model = self.create_single_gp(lower, upper)
        self.current_model.train(self.X, self.pred_y)
        self.iter = 1

    def train(self, X: np.ndarray, y: np.array):
        print('iter', self.iter)
        self.iter += 1
        self.update_incumbent(X, y)
        num_sample = y.shape[0]
        y = np.c_[y, np.array([self.historical_task_num] * num_sample)]

        # Train the ranker and output the rankings.
        X = np.r_[self.X, X]
        y = np.r_[self.y, y]
        rank_svm = RankSVM()
        rank_svm.fit(X, y)
        pred_y = rank_svm.predict(X)
        self.update_scaled_incumbent(pred_y[-num_sample:])

        # lower = np.amin(X, axis=0)
        # upper = np.amax(X, axis=0)
        # self.current_model = self.create_single_gp(lower, upper)
        do_optimize = True if self.iter % 2 == 0 else False
        self.current_model.train(X, pred_y, do_optimize=do_optimize)

    def predict(self, X: np.array):
        mu, var = self.current_model.predict(X)
        return mu, var

import math
import numpy as np
from tlbo.model.basics.se_kernel import SEKernel


class GPR(object):
    def __init__(self, optimize=True):
        self.kernel = SEKernel()
        self.L, self.alpha = None, None
        self.X, self.y = None, None
        self.optimize = optimize

    def train(self, X, y):
        self.X, self.y = X, y
        if self.optimize:
            self.kernel.optimize_hp(X, y)
        K = self.kernel.get_kernel_matrix(X)
        self.L = np.linalg.cholesky(K)
        t = np.linalg.solve(self.L, y)
        self.alpha = np.linalg.solve(self.L.T, t)

    def predict(self, X):
        res_mean, res_var = [], []
        for x in X:
            # Get k* vector.
            k_star = list()
            for i in range(self.X.shape[0]):
                k_star.append(self.kernel.get_kernel_value(x, self.X[i]))
            k_star = np.array(k_star)

            f_mean = np.dot(k_star, self.alpha)
            v = np.linalg.solve(self.L, k_star)
            f_var = self.kernel.get_kernel_value(x, x) - np.dot(v, v)
            res_mean.append(f_mean)
            res_var.append(f_var)
        return np.array(res_mean), np.array(res_var)

    def get_negative_log_likelihodd(self):
        log_determinant = 0.
        n = self.L.shape[0]
        for i in range(n):
            log_determinant += math.log(self.L[i][i])
        log_prob = -0.5*np.dot(self.y, self.alpha) - log_determinant - n/2.*math.log(2*math.pi)
        return log_prob

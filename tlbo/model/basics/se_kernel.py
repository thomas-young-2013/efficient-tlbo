import math
import scipy
import numpy as np
from scipy import optimize


class SEKernel(object):
    def __init__(self):
        self.use_gradients = False
        self.sigma_f, self.sigma_l, self.sigma_y = 1, 1, 1e-3

    def get_kernel_matrix(self, X, theta=None):
        if theta is None:
            sigma_f, sigma_l, sigma_y = self.sigma_f, self.sigma_l, self.sigma_y
        else:
            sigma_f, sigma_l, sigma_y = theta

        n = X.shape[0]
        diff_m = self.get_diff_matrix(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    K[i][j] = sigma_f*sigma_f + sigma_y*sigma_y
                else:
                    value = sigma_f*sigma_f*(math.exp(-0.5*diff_m[i][j]/(sigma_l*sigma_l)))
                    K[i][j] = value
                    K[j][i] = value
        return K

    def get_kernel_value(self, x1, x2):
        l2_diff = np.linalg.norm(x1-x2, 2)
        if l2_diff == 0:
            return self.sigma_f*self.sigma_f + self.sigma_y*self.sigma_y
        else:
            return self.sigma_f*self.sigma_f*(math.exp(-0.5*l2_diff/(self.sigma_l*self.sigma_l)))

    def optimize_hp(self, X, y):
        diff_mat = self.get_diff_matrix(X)

        def loglikelihood_f(x):
            K = self.get_kernel_matrix(X, theta=x)
            L = np.linalg.cholesky(K)
            t = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, t)

            log_determinant = 0.
            n = L.shape[0]
            for i in range(n):
                log_determinant += math.log(L[i][i])
            log_prob = -0.5 * np.dot(y, alpha) - log_determinant - n / 2. * math.log(2 * math.pi)

            # Turn it to minimization problem.
            return -log_prob

        def loglikelihood_f_der(x):
            der = np.zeros(len(x))
            K = self.get_kernel_matrix(X, theta=x)
            L = np.linalg.cholesky(K)
            t = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, t)

            _, l, u = scipy.linalg.lu(K)
            # fix_term = np.outer(alpha, alpha) - np.linalg.inv(K)
            fix_term = np.outer(alpha, alpha) - np.dot(np.linalg.inv(u), np.linalg.inv(l))
            n = K.shape[0]

            # Common part.
            def compute_der(der_m):
                trace = 0.
                for i in range(n):
                    trace += np.dot(fix_term[i], der_m[:, i])
                return 0.5*trace

            # Compute the derivative of sigma_y.
            tmp_m = np.zeros((n, n))
            for i in range(n):
                tmp_m[i][i] = 2*x[-1]
            der[-1] = compute_der(tmp_m)

            # compute the derivative of sigma_l.
            tmp_m = np.zeros((n, n))
            exp_term = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    exp_term[i][j] = math.exp(-0.5*diff_mat[i][j]/(x[1]*x[1]))
                    exp_term[j][i] = exp_term[i][j]
            for i in range(n):
                for j in range(i, n):
                    tmp_m[i][j] = x[0]*x[0]*exp_term[i][j]*diff_mat[i][j]/(x[1]*x[1]*x[1])
                    tmp_m[j][i] = tmp_m[i][j]
            der[1] = compute_der(tmp_m)

            # compute the derivative of sigma_f.
            tmp_m = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    tmp_m[i][j] = 2*x[0]*exp_term[i][j]
                    tmp_m[j][i] = tmp_m[i][j]
            der[0] = compute_der(tmp_m)
            return -1*der

        p0 = np.array([1, 1, 1e-3])
        if self.use_gradients:
            theta, _, _ = optimize.minimize(loglikelihood_f, p0,
                                            method="BFGS",
                                            jac=loglikelihood_f_der)
            self.sigma_f, self.sigma_l, self.sigma_y = theta
        else:
            try:
                results = optimize.minimize(loglikelihood_f, p0, method='L-BFGS-B')
                if not results.success:
                    self.sigma_f, self.sigma_l, self.sigma_y = p0
                else:
                    self.sigma_f, self.sigma_l, self.sigma_y = results.x
            except ValueError:
                raise ValueError("Could not find a valid hyperparameter configuration! Use initial configuration")

    def get_diff_matrix(self, X):
        n = X.shape[0]
        diff_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff_matrix[i][j] = np.linalg.norm(X[i]-X[j], 2)
                diff_matrix[j][i] = diff_matrix[i][j]
        return diff_matrix

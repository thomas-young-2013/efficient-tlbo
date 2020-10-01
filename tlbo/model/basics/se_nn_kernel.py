import math
import scipy
import numpy as np
from scipy import optimize


class SENNKernel(object):
    def __init__(self, metafeatures, ratio, split, max_nearest_neighbor):
        self.use_gradients = False
        self.sigma_f, self.sigma_l, self.sigma_y = 1, 1, 1e-3
        self.ratio = ratio
        self.max_nn = max_nearest_neighbor
        self.split_flag = split
        self.B = 30
        size = len(metafeatures)
        dist_m = np.zeros((size, size))
        for i in range(size):
            for j in range(i+1, size):
                dist_m[i][j] = np.linalg.norm(metafeatures[i]-metafeatures[j], 2)
                dist_m[j][i] = dist_m[i][j]

        # Build nearest neighbor lookuptable.
        self.lookup = dict()
        for i in range(size):
            id = str(metafeatures[i])
            indexs = np.argsort(dist_m[i])[:self.max_nn]
            self.lookup[id] = set()
            for index in indexs:
                self.lookup[id].add(str(metafeatures[index]))

    def get_kernel_matrix(self, X, theta=None):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i][j] = self.get_kernel_value(X[i], X[j], theta=theta)
                K[j][i] = K[i][j]
        return K

    def get_kernel_value(self, x1, x2, theta=None):
        if theta is None:
            sigma_f, sigma_l, sigma_y = self.sigma_f, self.sigma_l, self.sigma_y
        else:
            sigma_f, sigma_l, sigma_y = theta
        xq1, xq2 = x1[self.split_flag:],  x2[self.split_flag:]
        xp1, xp2 = x1[:self.split_flag],  x2[:self.split_flag]

        k2 = 1 - np.linalg.norm(x1 - x2, 2) / self.B
        assert k2 > 0
        # K nearest neighbors.
        # if str(xq2) in self.lookup[str(xq1)]:
        #     k2 = 1 - np.linalg.norm(x1-x2, 2)/self.B
        #     assert k2 > 0
        # else:
        #     k2 = 0

        # the SQ kernel works on hyperparaemter space.
        # l2_diff = np.linalg.norm(x1 - x2, 2)
        l2_diff = np.linalg.norm(xp1-xp2, 2)
        if l2_diff == 0:
            k1 = sigma_f*sigma_f + sigma_y*sigma_y
        else:
            k1 = sigma_f*sigma_f*(math.exp(-0.5*l2_diff/(sigma_l*sigma_l)))
        kernel_value = (1-self.ratio)*k1 + self.ratio*k2
        assert kernel_value >= 0
        return kernel_value

    def optimize_hp(self, X, y):
        diff_mat = self.get_diff_matrix(X)

        def loglikelihood_f(x):
            print('='*5, x)
            K = self.get_kernel_matrix(X, theta=x)
            # print(K)
            # print(np.linalg.eigvalsh(K))
            # TODO: for debug.
            L = scipy.linalg.cholesky(K, lower=True)
            # L = np.linalg.cholesky(K)
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
            return -1*(1-self.ratio)*der

        p0 = np.array([1, 1, 1e-3])
        if self.use_gradients:
            theta, _, _ = optimize.minimize(loglikelihood_f, p0,
                                            method="BFGS",
                                            jac=loglikelihood_f_der)
            self.sigma_f, self.sigma_l, self.sigma_y = theta
        else:
            try:
                results = optimize.minimize(loglikelihood_f, p0, method='L-BFGS-B')
                status = True if (results.x < 0).any() else False
                if not results.success or status:
                    self.sigma_f, self.sigma_l, self.sigma_y = p0
                else:
                    self.sigma_f, self.sigma_l, self.sigma_y = results.x
            except ValueError:
                raise ValueError("Could not find a valid hyperparameter configuration! Use initial configuration")

    def get_diff_matrix(self, X):
        X_ = X[:, :self.split_flag]
        n = X_.shape[0]
        diff_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff_matrix[i][j] = np.linalg.norm(X_[i]-X_[j], 2)
                diff_matrix[j][i] = diff_matrix[i][j]
        return diff_matrix

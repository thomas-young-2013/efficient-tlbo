import itertools
import numpy as np
from scipy.optimize import minimize


def scipy_solve(P, b, params, pred=None, debug=False):
    norm, lbd, mu, alpha = params
    A, t = P
    n, m = A.shape
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array(x),
                 'jac': lambda x: np.eye(len(x))}
    eq_cons = {'type': 'eq',
                 'fun': lambda x: np.array([sum(x) - 1]),
                 'jac': lambda x: np.array([1.]*len(x))}
    if pred is not None:
        B = pred[0]
        p = pred[1]
        group_size = B.shape[0] // m
    x0 = np.array([1. / m] * m)
    scale0 = 1./n
    scale1 = 1.

    def f(x):
        ls_value = scale0 * np.linalg.norm(alpha*(A*np.mat(x).T) + (1 - alpha)*t - b, 2) + lbd * np.linalg.norm(x, norm)
        if pred is None:
            return ls_value
        else:
            generalized_error = 0.
            for i in range(m):
                start_index = i * group_size
                end_index = start_index + group_size
                gen_error = (scale1 * x[i] * np.linalg.norm(
                    B[start_index: end_index] * np.mat(x).T - p[start_index: end_index], 2))
                generalized_error += gen_error
            return ls_value + mu*generalized_error

    def f_der(x):
        ls_der = 2. * scale0 * np.array(alpha*A.T*(alpha*(A*np.mat(x).T) + (1 - alpha)*t - b))[:, 0] +\
                 (2*lbd*x if norm == 2 else lbd*np.sign(x))
        if pred is None:
            return ls_der
        else:
            gen_der = np.zeros(m)
            for s in range(m):
                start_index = s * group_size
                end_index = start_index + group_size
                B_t = B[start_index: end_index]
                p_t = p[start_index: end_index]
                w = np.mat(x).T
                s_der = 2 * x[s] * scale1 * B_t.T * (B_t*w - p_t)
                assert s_der.shape == (m, 1)
                s_der = s_der.A1
                s_der[s] += scale1 * np.linalg.norm(B_t*w - p_t, 2)
                gen_der += s_der
            return ls_der + mu*gen_der
    res = minimize(f, x0, method='SLSQP', jac=f_der, constraints=[eq_cons, ineq_cons], options={'ftol': 1e-8, 'disp': False})

    def f_target(x):
        l1_value = scale0 * np.linalg.norm(alpha*(A*np.mat(x).T) + (1 - alpha)*t - b, 2)
        l2_value = lbd * np.linalg.norm(x, norm)
        if pred is None:
            return l1_value, l2_value, l1_value + l2_value
        else:
            generalized_error = 0.
            for i in range(m):
                start_index = i * group_size
                end_index = start_index + group_size
                generalized_error += (scale1 * x[i] * np.linalg.norm(
                    B[start_index: end_index] * np.mat(x).T - p[start_index: end_index], 2))
            return l1_value, l2_value, mu*generalized_error, l1_value + l2_value + mu*generalized_error

    status = False if np.isnan(res.x).any() else True
    if not res.success and status:
        res.x[res.x < 0.] = 0.
        res.x[res.x > 1.] = 1.
    # debug = True
    if debug:
        print(f_target(res.x))
    return res.x, status


def L_rank(true_y, pred_y, func_id):
    true_y = np.array(true_y)[:, 0]
    pred_y = np.array(pred_y)[:, 0]
    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Precompute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    loss = 0.
    for (i, j) in pairs:
        if func_id == 1:
            loss += max(pred_y[j] - pred_y[i], 0.)
        elif func_id == 2:
            loss += np.exp(pred_y[j] - pred_y[i])
        elif func_id == 3:
            loss += np.log(1 + np.exp(pred_y[j] - pred_y[i]))
        else:
            raise ValueError('Invalid func id!')
    return loss


def L_der(true_y, A, x, t, alpha, func_id):
    true_y = np.array(true_y)[:, 0]
    if t is None:
        pred_y = A * np.mat(x).T
    else:
        pred_y = alpha*A*np.mat(x).T + (1-alpha)*t
    pred_y = np.array(pred_y)[:, 0]

    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Precompute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    grad = np.zeros(A.shape[1])
    for (i, j) in pairs:
        if func_id == 1:
            if pred_y[j] > pred_y[i]:
                grad += (A[j] - A[i]).A1
        elif func_id == 2:
            grad += np.exp(pred_y[j] - pred_y[i]) * (A[j] - A[i]).A1
        elif func_id == 3:
            e_z = np.exp(pred_y[j] - pred_y[i])
            grad += e_z / (1 + e_z) * (A[j] - A[i]).A1
        else:
            raise ValueError('Invalid func id!')
    if t is not None:
        grad *= alpha
    return grad


def scipy_solve_rank(P, b, params, pred=None, debug=False):
    norm, lbd, mu, alpha, func_id = params
    A, t = P
    n, m = A.shape
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array(x),
                 'jac': lambda x: np.eye(len(x))}
    eq_cons = {'type': 'eq',
               'fun': lambda x: np.array([sum(x) - 1]),
               'jac': lambda x: np.array([1.] * len(x))}
    if pred is not None:
        B = pred[0]
        p = pred[1]
        group_size = B.shape[0] // m
    x0 = np.array([1. / m] * m)
    scale0 = 1./n
    scale1 = 1.

    def f(x):
        ls_value = scale0 * L_rank(b, alpha * A * np.mat(x).T + (1-alpha)*t, func_id) + lbd * np.linalg.norm(x, norm)
        if pred is None:
            return ls_value
        else:
            generalized_error = 0.
            for i in range(m):
                start_index = i * group_size
                end_index = start_index + group_size
                generalized_error += (scale1 * x[i] * L_rank(p[start_index: end_index],
                                                                 B[start_index: end_index] * np.mat(x).T, func_id))
            return ls_value + mu * generalized_error

    def f_der(x):
        ls_der = scale0 * L_der(b, A, x, t, alpha, func_id) + (2 * lbd * x if norm == 2 else lbd * np.sign(x))
        if pred is None:
            return ls_der
        else:
            gen_der = np.zeros(m)
            for s in range(m):
                start_index = s * group_size
                end_index = start_index + group_size
                B_t = B[start_index: end_index]
                p_t = p[start_index: end_index]
                w_t = np.mat(x).T
                s_der = scale1 * x[s] * L_der(p_t, B_t, x, None, alpha, func_id)
                assert s_der.shape[0] == m
                s_der[s] += scale1 * L_rank(p_t, B_t*w_t, func_id)
                gen_der += s_der
            return ls_der + mu * gen_der

    res = minimize(f, x0, method='SLSQP', jac=f_der, constraints=[eq_cons, ineq_cons],
                   options={'ftol': 1e-8, 'disp': False})

    def f_target(x):
        l1_value = scale0 * L_rank(b, alpha * A * np.mat(x).T + (1-alpha)*t, func_id)
        l2_value = lbd * np.linalg.norm(x, norm)
        if pred is None:
            return l1_value, l2_value, l1_value + l2_value
        else:
            generalized_error = 0.
            for i in range(m):
                start_index = i * group_size
                end_index = start_index + group_size
                generalized_error += (scale1 * x[i] * L_rank(p[start_index: end_index],
                                                                 B[start_index: end_index] * np.mat(x).T, func_id))
            return l1_value, l2_value, mu * generalized_error, l1_value + l2_value + mu * generalized_error

    status = False if np.isnan(res.x).any() else True
    if not res.success and status:
        res.x[res.x < 0.] = 0.
        res.x[res.x > 1.] = 1.
    debug = True
    if debug:
        print(f_target(res.x))
    return res.x, status

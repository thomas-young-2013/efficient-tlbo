import itertools
import numpy as np
from scipy.optimize import minimize


def scipy_solve(A, b, params, B, debug=False):
    norm, lbd, mu = params
    n, m = A.shape

    # Add constraints.
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array(x),
                 'jac': lambda x: np.eye(len(x))}
    eq_cons = {'type': 'eq',
                 'fun': lambda x: np.array([sum(x) - 1]),
                 'jac': lambda x: np.array([1.]*len(x))}

    # group_size = (B.shape[0]-n) // (m - 1)
    group_size = B.shape[0]
    normalize_scalar = 1./group_size
    x0 = np.array([1. / m] * m)

    def f(x):
        w = np.mat(x).T
        loss = 1./n * np.linalg.norm(A*w - b, 2) + lbd * np.linalg.norm(w, norm)
        # Add data-dependent regularizer.
        loss_reg = 0.
        for i in range(m):
            # normalize_scalar = (1./group_size) if i < m - 1 else 1./n
            loss_reg += normalize_scalar * x[i] * np.linalg.norm(B*w - B[:, i], 2)
        loss += mu*loss_reg
        return loss

    def f_der(x):
        w = np.mat(x).T
        der = 2./n * np.array(A.T*(A*w - b))[:, 0] + lbd*(2*x if norm == 2 else np.sign(x))
        reg_der = np.zeros(m)
        for i in range(m):
            # normalize_scalar = (1./group_size) if i < m - 1 else 1./n
            s_der = 2*x[i]*normalize_scalar * B.T * (B*w - B[:, i])
            assert s_der.shape == (m, 1)
            s_der = s_der.A1
            s_der[i] += normalize_scalar * np.linalg.norm(B*w - B[:, i], 2)
            reg_der += s_der
        der += mu * reg_der
        return der
    res = minimize(f, x0, method='SLSQP', jac=f_der, constraints=[eq_cons, ineq_cons], options={'ftol': 1e-8, 'disp': False})

    # Display the loss value for debugging.
    def f_val(x):
        w = np.mat(x).T
        loss = 1./n * np.linalg.norm(A*w - b, 2)
        loss_reg1 = lbd * np.linalg.norm(w, norm)
        loss_reg2 = 0.
        for i in range(m):
            # normalize_scalar = (1./group_size) if i < m - 1 else 1./n
            loss_reg2 += normalize_scalar * x[i] * np.linalg.norm(B*w - B[:, i], 2)
        loss_reg2 *= mu
        return loss, loss_reg1, loss_reg2, loss+loss_reg1+loss_reg2

    status = False if np.isnan(res.x).any() else True
    if not res.success and status:
        res.x[res.x < 0.] = 0.
        res.x[res.x > 1.] = 1.
        if sum(res.x) > 1.5:
            status = False

    if debug:
        print(f_val(res.x))
    return res.x, status


def L_rank(true_y, pred_y, func_id):
    true_y = np.array(true_y)[:, 0]
    pred_y = np.array(pred_y)[:, 0]
    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Compute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    loss = 0.
    pair_num = len(pairs)
    if pair_num == 0:
        return 0.
    for (i, j) in pairs:
        if func_id == 1:
            loss += max(pred_y[j] - pred_y[i], 0.)
        elif func_id == 2:
            loss += np.exp(pred_y[j] - pred_y[i])
        elif func_id == 3:
            loss += np.log(1 + np.exp(pred_y[j] - pred_y[i]))
        else:
            raise ValueError('Invalid loss type!')
    return loss/pair_num


def L_der(true_y, A, x, func_id):
    true_y = np.array(true_y)[:, 0]
    pred_y = np.array(A*np.mat(x).T)[:, 0]

    comb = itertools.combinations(range(true_y.shape[0]), 2)
    pairs = list()
    # Compute the pairs.
    for _, (i, j) in enumerate(comb):
        if true_y[i] > true_y[j]:
            pairs.append((i, j))
        elif true_y[i] < true_y[j]:
            pairs.append((j, i))
    # Calculate the derivatives.
    grad = np.zeros(A.shape[1])
    pair_num = len(pairs)
    if pair_num == 0:
        return grad
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
    return grad/pair_num


def scipy_solve_rank(A, b, params, comb, debug=False):
    norm, lbd, mu_type, loss_type = params
    n, m = A.shape
    B, ys = comb
    ns = B.shape[0]
    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: np.array(x),
                 'jac': lambda x: np.eye(len(x))}
    eq_cons = {'type': 'eq',
               'fun': lambda x: np.array([sum(x) - 1]),
               'jac': lambda x: np.array([1.] * len(x))}
    assert ns % m == 0
    b_size = int(ns/m)
    x0 = np.array([1. / m] * m)
    # mu = 1. - n/b_size
    # mu = 0.2 if mu < 0.2 else mu
    mu = 0.5

    def f(x):
        w = np.mat(x).T
        loss = (1-mu) * L_rank(b, A*w, loss_type) + lbd * np.linalg.norm(w, norm)
        da_reg = 0.
        for i in range(m):
            s_index = b_size*i
            e_index = s_index + b_size
            da_reg += x[i] * L_rank(ys[s_index:e_index], B[s_index:e_index]*w, loss_type)
        loss += mu*da_reg
        return loss

    def f_der(x):
        w = np.mat(x).T
        der = (1-mu) * L_der(b, A, x, loss_type) + lbd*(2*x if norm == 2 else np.sign(x))
        der_reg = np.zeros(m)
        for i in range(m):
            s_index = b_size * i
            e_index = s_index + b_size
            y_s = ys[s_index:e_index]
            B_s = B[s_index:e_index]
            s_der = x[i] * L_der(y_s, B_s, x, loss_type)
            assert s_der.shape[0] == m
            s_der[i] += L_rank(y_s, B_s*w, loss_type)
            der_reg += s_der
        der += mu*der_reg
        return der

    res = minimize(f, x0, method='SLSQP', jac=f_der, constraints=[eq_cons, ineq_cons],
                   options={'ftol': 1e-8, 'disp': False})

    # Display the loss value for debugging.
    def f_val(x):
        w = np.mat(x).T
        loss = (1-mu) * L_rank(b, A*w, loss_type)
        loss_reg1 = lbd * np.linalg.norm(w, norm)
        loss_reg2 = 0.
        for i in range(m):
            s_index = b_size * i
            e_index = s_index + b_size
            loss_reg2 += x[i] * L_rank(ys[s_index:e_index], B[s_index:e_index] * w, loss_type)
        loss_reg2 *= mu
        return loss, loss_reg1, loss_reg2, loss+loss_reg1+loss_reg2

    status = False if np.isnan(res.x).any() else True
    if not res.success and status:
        res.x[res.x < 0.] = 0.
        res.x[res.x > 1.] = 1.
        if sum(res.x) > 1.5:
            status = False

    if debug:
        print(f_val(res.x))
    return res.x, status

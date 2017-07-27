"""Converts different optimization problems types"""

import numpy as np
import scipy.sparse as spc

__all__ = ['parse_cutest_like_problem',
           'parse_matlab_like_problem']

xc = None
c = None
xa = None
A = None


def parse_cutest_like_problem(
        fun, grad, lagr_hess, n_vars, constr, jac,
        c_lb, c_ub, lb, ub, eq_list=None):
    """Converts CUTEst-like constraints to general format.

    Converts a optimization problem of the form:

        minimize fun(x)
        subject to: c_lb <= constr(x) <= c_ub
                    lb <= x <= ub

    to an equivalent optimization problem:

        minimize fun(x)
        subject to: g(x) <= 0
                    h(x) = 0
    """
    n_constr, = np.shape(c_lb)

    # Get list of equality constraints
    # by looking at the values for which
    # c_lb[i]== c_ub[i]. Only does that
    # when no list is provided.
    if eq_list is None:
        eq_list = np.zeros(n_constr, dtype=bool)
        for i in range(n_constr):
            if not np.isinf(c_lb[i]) and c_lb[i] == c_ub[i]:
                eq_list[i] = True

    # Get positions c_lb and c_ub are finite
    c_lb_finite = ~np.isinf(c_lb) & ~eq_list
    c_ub_finite = ~np.isinf(c_ub) & ~eq_list
    n_c_lb = np.int(np.sum(c_lb_finite))
    n_c_ub = np.int(np.sum(c_ub_finite))
    # Get number of equality constraints
    n_eq = np.int(np.sum(eq_list))
    n_ineq = n_c_lb + n_c_ub
    # Compute number of finite lower and upper bounds
    lb_finite = ~np.isinf(lb)
    ub_finite = ~np.isinf(ub)
    n_lb = np.int(np.sum(lb_finite))
    n_ub = np.int(np.sum(ub_finite))
    # Total number of constraints
    n_total_eq = n_eq
    n_total_ineq = n_ineq + n_lb + n_ub

    def constr_ineq(x):
        # Scheme for avoiding evaluate the constraints
        # multiple times for the same value of x.
        global xc, c
        if not np.array_equal(xc, x):
            c = constr(x)
            xc = x

        return np.hstack((c[c_ub_finite] - c_ub[c_ub_finite],
                          c_lb[c_lb_finite] - c[c_lb_finite],
                          x[ub_finite] - ub[ub_finite],
                          lb[lb_finite] - x[lb_finite]))

    def constr_eq(x):
        # Scheme for avoiding evaluate the constraints
        # multiple times for the same value of x.
        global xc, c
        if not np.array_equal(xc, x):
            c = constr(x)
            xc = x

        return c[eq_list] - c_ub[eq_list]

    def jac_ineq(x):
        # Scheme for avoiding evaluate the Jacobian
        # matrix multiple times for the same value of x.
        global xa, A
        if not np.array_equal(xa, x):
            A = jac(x)
            xa = x

        I = spc.eye(n_vars).tocsc()
        I_ub = I[ub_finite, :]
        I_lb = I[lb_finite, :]
        return spc.vstack([A[c_ub_finite, :],
                           -A[c_lb_finite, :],
                           I_ub,
                           -I_lb])

    def jac_eq(x):
        # Scheme for avoiding evaluate the Jacobian
        # matrix multiple times for the same value of x.
        global xa, A
        if not np.array_equal(xa, x):
            A = jac(x)
            xa = x

        return A[eq_list, :]

    def new_lagr_hess(x, v_eq, v_ineq):
        v = np.zeros(n_constr)
        v[eq_list] = v_eq
        v[c_ub_finite] = v_ineq[:n_c_ub]
        v[c_lb_finite] -= v_ineq[n_c_ub:n_c_ub+n_c_lb]
        return lagr_hess(x, v)

    box_constraints = np.hstack((np.zeros(n_ineq, dtype=bool),
                                np.ones(n_lb+n_ub, dtype=bool)))

    return fun, grad, new_lagr_hess, n_total_ineq, constr_ineq, \
        jac_ineq, n_total_eq, constr_eq, jac_eq, box_constraints


def parse_matlab_like_problem(
        fun, grad, lagr_hess, n_vars, n_eq=0, n_ineq=0, constr_ineq=None,
        jac_ineq=None, constr_eq=None, jac_eq=None, A_ineq=None, b_ineq=None,
        A_eq=None, b_eq=None, lb=None, ub=None):
    """Converts Matlab-like constraints to general format.

    Converts an optimization problem of the form:

        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
                    A_ineq x <= b_ineq
                    A_eq x = b_eq
                    lb <= x <= ub

    to an equivalent optimization problem:

        minimize fun(x)
        subject to: new_constr_ineq(x) <= 0
                    new_constr_eq(x) = 0
    """
    # Define empty default functions
    empty_vec = np.empty((0,))
    empty_matrix = np.empty((0, n_vars))

    def empty_constr(x):
        return empty_vec

    def empty_jac(x):
        return empty_matrix

    # Fill with empty arrays when needed
    constr_ineq = constr_ineq if constr_ineq is not None else empty_constr
    jac_ineq = jac_ineq if jac_ineq is not None else empty_jac
    constr_eq = constr_eq if constr_eq is not None else empty_constr
    jac_eq = jac_eq if jac_eq is not None else empty_jac
    A_ineq = A_ineq if A_ineq is not None else empty_matrix
    b_ineq = b_ineq if b_ineq is not None else empty_vec
    A_eq = A_eq if A_eq is not None else empty_matrix
    b_eq = b_eq if b_eq is not None else empty_vec
    lb = lb if lb is not None else np.full((n_vars,), -np.inf)
    ub = ub if ub is not None else np.full((n_vars,), np.inf)

    # Get dimensions of linear constraints
    n_lin_eq, = np.shape(b_eq)
    n_lin_ineq, = np.shape(b_ineq)
    # Compute number of finite lower and upper bounds
    lb_finite = ~np.isinf(lb)
    n_lb = np.int(np.sum(lb_finite))
    ub_finite = ~np.isinf(ub)
    n_ub = np.int(np.sum(ub_finite))
    # Total number of constraints
    n_total_eq = n_eq + n_lin_eq
    n_total_ineq = n_ineq + n_lin_ineq + n_lb + n_ub

    def new_constr_ineq(x):
        return np.hstack((constr_ineq(x),
                          A_ineq.dot(x) - b_ineq,
                          x[ub_finite] - ub[ub_finite],
                          lb[lb_finite] - x[lb_finite]))

    def new_constr_eq(x):
        return np.hstack((constr_eq(x),
                          A_eq.dot(x) - b_eq))

    def new_jac_ineq(x):
        I = spc.eye(n_vars).tocsc()
        I_ub = I[ub_finite, :]
        I_lb = I[lb_finite, :]

        if n_total_ineq > 0:
            return spc.vstack([jac_ineq(x),
                               A_ineq,
                               I_ub,
                               -I_lb])
        else:
            return empty_matrix

    def new_jac_eq(x):
        if n_total_eq > 0:
            return spc.vstack([jac_eq(x),
                               A_eq])
        else:
            return empty_matrix

    def new_lagr_hess(x, v_eq, v_ineq):
        return lagr_hess(x, v_eq[:n_eq], v_ineq[:n_ineq])

    box_constraints = np.hstack((np.zeros(n_ineq + n_lin_ineq, dtype=bool),
                                np.ones(n_lb+n_ub, dtype=bool)))

    return fun, grad, new_lagr_hess, n_total_ineq, new_constr_ineq, \
        new_jac_ineq, n_total_eq, new_constr_eq, new_jac_eq, box_constraints

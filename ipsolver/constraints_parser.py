"""Converts different optimization problems types"""

import numpy as np
import scipy.sparse as spc

__all__ = ['parse_linear_constraints',
           'parse_nonlinear_constraints']


def parse_linear_constraints(A, lb, ub):
    """Converts linear inequality constraints.

    Converts:
        lb <= A x <= ub
    to:
        A_ineq x <= b_ineq
    """
    # Get positions lb and ub are finite
    lb_finite = np.invert(np.isinf(lb))
    ub_finite = np.invert(np.isinf(ub))
    # Assemble ``b_ineq``
    b_ineq = np.hstack((ub[ub_finite], -lb[lb_finite]))
    # Assemble ``A_ineq``
    if spc.issparse(A):
        A_ineq = spc.vstack((A[ub_finite, :], -A[lb_finite, :]))
    else:
        A_ineq = np.vstack((A[ub_finite, :], -A[lb_finite, :]))
    return A_ineq, b_ineq


def parse_nonlinear_constraints(constraint, jacobian, lb, ub):
    """Converts nonlinear inequality constraints.

    Converts:
        lb <= constraint(x) <= ub
    to:
        constr_ineq(x) <= 0
    """
    # Get positions lb and ub are finite
    lb_finite = np.invert(np.isinf(lb))
    ub_finite = np.invert(np.isinf(ub))

    def constr_ineq(x):
        c = constraint(x)
        return np.hstack((c[ub_finite] - ub[ub_finite],
                          lb[lb_finite] - c[lb_finite]))

    def jac_ineq(x):
        A = jacobian(x)
        # Assemble ``A_ineq``
        if spc.issparse(A):
            A_ineq = spc.vstack((A[ub_finite, :], -A[lb_finite, :]))
        else:
            A_ineq = np.vstack((A[ub_finite, :], -A[lb_finite, :]))
        return A_ineq

    return constr_ineq, jac_ineq

"""
Equality-constrained quadratic programming solvers.
"""

from __future__ import division, print_function, absolute_import
from scipy.sparse import linalg, bmat, csc_matrix
import numpy as np

__all__ = [
    'eqp_kktfact'
]


def eqp_kktfact(G, c, A, b):
    """
    Solve equality-constrained quadratic programming (EQP) problem
    ``min 1/2 x.T G x + x.t c``  subject to ``A x = b``
    using direct factorization of the KKT system.

    Parameters
    ----------
    G, A : sparse matrix
        Hessian and Jacobian matrix of the EQP
    c, b : ndarray
        Unidimensional arrays.

    Returns
    -------
    x : ndarray
        Solution of the KKT problem
    lagrange_multipliers : ndarray
        Lagrange multipliers of the KKT problem
    """

    n = len(c)  # Number of parameters
    m = len(b)  # Number of constraints

    # Karush-Kuhn-Tucker matrix of coeficients.
    # Defined as in Nocedal/Wright "Numerical
    # Optimization" p.452 in Eq. (16.4)
    kkt_matrix = csc_matrix(bmat([[G, A.T], [A, None]]))

    # Vector of coeficients.
    kkt_vec = np.hstack([-c, b])

    # TODO: Use a symmetric indefinite factorization
    #       to solve the system twice as fast (because
    #       of the symmetry)
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)

    x = kkt_sol[:n]
    lagrange_multipliers = -kkt_sol[n:n+m]

    return x, lagrange_multipliers

"""
Equality-constrained quadratic programming solvers.
"""

from __future__ import division, print_function, absolute_import
from scipy.sparse import (linalg, bmat, csc_matrix, eye, issparse,
                          isspmatrix_csc, isspmatrix_csr)
from scipy.sparse.linalg import LinearOperator
from sksparse.cholmod import cholesky_AAt
import numpy as np

__all__ = [
    'eqp_kktfact',
    'projections',
    'projected_cg',
    'orthogonality'
]


# For comparison with the projected CG
def eqp_kktfact(H, c, A, b):
    """
    Solve equality-constrained quadratic programming (EQP) problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x = b``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H, A : sparse matrix
        Hessian and Jacobian matrix of the EQP problem.
    c, b : ndarray


        Unidimensional arrays.

    Returns
    -------
    x : ndarray
        Solution of the KKT problem.
    lagrange_multipliers : ndarray
        Lagrange multipliers of the KKT problem.
    """

    n = len(c)  # Number of parameters
    m = len(b)  # Number of constraints

    # Karush-Kuhn-Tucker matrix of coeficients.
    # Defined as in Nocedal/Wright "Numerical
    # Optimization" p.452 in Eq. (16.4)
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))

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


def orthogonality(A, g):
    """
    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given
    vector ``g``:
    ``orth =  norm(A g)/(norm(A)*norm(g))``.
    The formula is a more efficient version of
    formula (3.13) from [1]_.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """

    # Compute vector norms
    norm_g = np.linalg.norm(g)

    # Compute frobenius norm of the matrix A
    if issparse(A):
        norm_A  = linalg.norm(A, ord='fro') 
    else:
        norm_A = np.linalg.norm(A, ord='fro')  

    # Check if norms are zero
    if norm_g == 0 or norm_A == 0:
        return 0

    norm_A_g = np.linalg.norm(A.dot(g))

    # Orthogonality measure
    orth = norm_A_g/(norm_A*norm_g)

    return orth


def projections(A, method='NormalEquation'):
    """
    Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : sparse matrix
         Matrix ``A`` used in the projection.
    method : string
        Method used for compute the given linear
        operators. Should be one of:

            - 'NormalEquation': The operators
               will be computed using the
               so-called normal equation approach
               explained in [1]_.
               In order to do so the Cholesky
               factorization of ``(A A.T)`` is
               computed using CHOLMOD.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_ p.463.
    Returns
    -------
    Z : LinearOperator
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator
        Least-Square operator. For a given vector ``x``,
        the least-square operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """

    # Iterative Refinement Parameters
    ORTH_TOLERANCE = 1e-12
    MAX_INTERACTIONS = 3

    m, n = np.shape(A)

    if method == 'NormalEquation':

        # Cholesky factorization
        factor = cholesky_AAt(A)

        # z = x - A.T inv(A A.T) A x
        def null_space(x):

            v = factor(A.dot(x))
            z = x - A.T.dot(v)

            # Iterative refinement to improve roundoff
            # errors described in [2]_, algorithm 5.1
            k = 0
            while orthogonality(A, z) > ORTH_TOLERANCE:
                # z_next = z - A.T inv(A A.T) A z
                v = factor(A.dot(z))
                z = z - A.T.dot(v)
                k += 1
                if k > MAX_INTERACTIONS:
                    break

            return z

        # z = inv(A A.T) A x
        def least_squares(x):
            return factor(A.dot(x))

        # z = A.T inv(A A.T) x
        def row_space(x):
            return A.T.dot(factor(x))

    elif method == 'AugmentedSystem':

        # Form aumengted system
        K = csc_matrix(bmat([[eye(n), A.T], [A, None]]))

        # LU factorization
        # TODO: Use a symmetric indefinite factorization
        #       to solve the system twice as fast (because
        #       of the symmetry)
        factor = linalg.splu(K)

        # z = x - A.T inv(A A.T) A x
        # is computed solving the extended system:
        # [I A.T] * [ z ] = [x]
        # [A  O ]   [aux]   [0]
        def null_space(x):
            # v = [x]
            #     [0]
            v = np.hstack([x, np.zeros(m)])

            # lu_sol = [ z ]
            #          [aux]
            lu_sol = factor.solve(v)

            z = lu_sol[:n]

            # Iterative refinement to improve roundoff
            # errors described in [2]_, algorithm 5.2
            k = 0
            while orthogonality(A, z) > ORTH_TOLERANCE:
                # new_v = [x] - [I A.T] * [ z ]
                #         [0]   [A  O ]   [aux]
                new_v = v - K.dot(lu_sol)

                # [I A.T] * [delta  z ] = new_v
                # [A  O ]   [delta aux]
                lu_update = factor.solve(new_v)

                #  [ z ] += [delta  z ]
                #  [aux]    [delta aux]
                lu_sol = lu_sol + lu_update

                z = lu_sol[:n]
                k += 1
                if k > MAX_INTERACTIONS:
                    break

            # return z = x - A.T inv(A A.T) A x
            return z

        # z = inv(A A.T) A x
        # is computed solving the extended system:
        # [I A.T] * [aux] = [x]
        # [A  O ]   [ z ]   [0]
        def least_squares(x):
            # v = [x]
            #     [0]
            v = np.hstack([x, np.zeros(m)])

            # lu_sol = [aux]
            #          [ z ]
            lu_sol = factor.solve(v)

            # return z = inv(A A.T) A x
            return lu_sol[n:m+n]

        # z = A.T inv(A A.T) x
        # is computed solving the extended system:
        # [I A.T] * [ z ] = [0]
        # [A  O ]   [aux]   [x]
        def row_space(x):
            # v = [0]
            #     [x]
            v = np.hstack([np.zeros(n), x])

            # lu_sol = [ z ]
            #          [aux]
            lu_sol = factor.solve(v)

            # return z = A.T inv(A A.T) x
            return lu_sol[:n]

    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)

    return Z, LS, Y


def projected_cg(H, c, Z, Y, b, tol=None, return_all=False, max_iter=None):
    """
    Solve equality-constrained quadratic programming (EQP) problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x = b``
    using projected cg method.

    Parameters
    ----------
    H : LinearOperator, sparse matrix, ndarray
        Operator for computing ``H v``.
    c : ndarray
        Unidimensional array.
    Z : LinearOperator, sparse matrix, ndarray
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray
        Operator that, for a given a vector ``b``, compute a solution of
        of ``A x = b``.
    b : ndarray
        Unidimensional array.
    tol : float
        Tolerance used to interrupt the algorithm.
    max_inter : int
        Maximum algorithm iteractions. Where ``max_inter <= n-m``. By default
        uses ``max_iter = n-m``.
    return_all : bool
        When true return the list of all vectors through the iterations.

    Returns
    -------
    x : ndarray
        Solution of the KKT problem
    info : Dict
        Dictionary containing the following:

            - niter : Number of iteractions.
            - step_norm : gives the norm of the last step ``d`` scalated
                          by the projection matrix ``Z``, that is: ``d.T Z d``.
            - allvecs : List containing all intermediary vectors (optional).

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """

    n = len(c)  # Number of parameters
    m = len(b)  # Number of constraints

    # Initial Values
    x = Y.dot(b)
    r = Z.dot(H.dot(x) + c)
    g = Z.dot(r)
    p = -g

    # Store ``x`` value
    if return_all:
        allvecs = [x]

    # Values for the first iteration
    H_p = H.dot(p)
    rt_g = r.dot(g)

    # Set default tolerance
    if tol is None:
        tol = max(0.01 * np.sqrt(rt_g), 1e-12)

    # Set maximum iteractions
    if max_iter is None:
        max_iter = n-m
    max_iter = min(max_iter, n-m)

    k = 1
    for i in range(max_iter):

        # Stop Criteria r.T g < tol
        if rt_g < tol:
            break

        # Compute next step
        pt_H_p = H_p.dot(p)
        alpha = rt_g / pt_H_p
        x = x + alpha*p

        # Store ``x`` value
        if return_all:
            allvecs.append(x)

        # Update residual
        r_next = r + alpha*H_p

        # Project residual g+ = Z r+
        g_next = Z.dot(r_next)

        # Compute conjugate direction step d
        rt_g_next = r_next.dot(g_next)
        beta = rt_g_next / rt_g
        p = - g_next + beta*p

        # Prepare for next iteration
        g = g_next
        r = g_next
        rt_g = r.dot(g)
        H_p = H.dot(p)
        k += 1

    info = {'niter': k, 'step_norm': rt_g}
    if return_all:
        info['allvecs'] = allvecs

    return x, info

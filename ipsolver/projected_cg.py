"""
Equality-constrained quadratic programming solvers.
"""

from __future__ import division, print_function, absolute_import
from scipy.sparse import (linalg, bmat, csc_matrix, eye, issparse,
                          isspmatrix_csc, isspmatrix_csr)
from scipy.sparse.linalg import LinearOperator
from sksparse.cholmod import cholesky_AAt
from math import copysign
import numpy as np

__all__ = [
    'eqp_kktfact',
    'get_boundaries_intersections',
    'orthogonality',
    'projections',
    'projected_cg'
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


def get_boundaries_intersections(z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius**2
        sqrt_discriminant = np.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically
        # equivalent to:
        # ta = (-b - sqrt_discriminant) / (2*a)
        # tb = (-b + sqrt_discriminant) / (2*a)
        # but produce smaller round off errors.
        # Look at Matrix Computation p.97
        # for a better justification.
        aux = b + copysign(sqrt_discriminant, b)
        ta = -aux / (2*a)
        tb = -2*c / aux
        return sorted([ta, tb])


def orthogonality(A, g):
    """
    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given
    vector ``g``:
    ``orth =  norm(A g)/(norm(A)*norm(g))``.
    The formula is a more cheaper (and simplified) version of
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
        norm_A = linalg.norm(A, ord='fro')
    else:
        norm_A = np.linalg.norm(A, ord='fro')

    # Check if norms are zero
    if norm_g == 0 or norm_A == 0:
        return 0

    norm_A_g = np.linalg.norm(A.dot(g))

    # Orthogonality measure
    orth = norm_A_g/(norm_A*norm_g)

    return orth


def projections(A, method='NormalEquation', orth_tol=1e-12, max_refin=3):
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
               factorization of ``(A A.T)``
               is computed.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_.

    orth_tol : float
        Tolerance for iterative refinements.
    max_refin : int
        Maximum number of iterative refinements

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

    Notes
    -----
    Uses iterative refinements described in [1]
    during the computation of ``Z`` in order to
    cope with the possibility of large roundoff errors.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """

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
            while orthogonality(A, z) > orth_tol:
                if k >= max_refin:
                    break

                # z_next = z - A.T inv(A A.T) A z
                v = factor(A.dot(z))
                z = z - A.T.dot(v)
                k += 1

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
            while orthogonality(A, z) > orth_tol:
                if k >= max_refin:
                    break

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


def projected_cg(H, c, Z, Y, b, trust_radius=np.inf,
                 tol=None, return_all=False, max_iter=None):
    """
    Solve equality-constrained quadratic programming (EQP) problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x = b`` and
    to ``||x|| < trust_radius`` using projected cg method.

    Parameters
    ----------
    H : LinearOperator, sparse matrix or ndarray
        Operator for computing ``H v``.
    c : ndarray
        Unidimensional array.
    Z : LinearOperator, sparse matrix or ndarray
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray
        Operator that, for a given a vector ``b``, compute a solution of
        of ``A x = b``.
    b : ndarray
        Unidimensional array.
    trust_radius : float
        Trust radius to be considered. By default uses ``trust_radius=inf``,
        which means no trust radius at all.
    tol : float
        Tolerance used to interrupt the algorithm.
    max_inter : int
        Maximum algorithm iteractions. Where ``max_inter <= n-m``. By default
        uses ``max_iter = n-m``.
    return_all : bool
        When ``true`` return the list of all vectors through the iterations.

    Returns
    -------
    x : ndarray
        Solution of the KKT problem
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iteractions.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Termination from trust-region bound;
                3. Negative curvature detected;
                4. tolerance was satisfied.
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

    # If x > trust-region the problem does not have a solution
    tr_distance = trust_radius - np.linalg.norm(x)
    if tr_distance < 0:
        raise ValueError("Trust region problem does not have a solution.")

    # If x == trust_radius, then x is the solution
    # to the optimization problem, since x is the
    # minimum norm solution to Ax=b
    elif tr_distance < 1e-12:
        hits_boundary = True
        info = {'niter': 0, 'stop_cond': 2}
        if return_all:
            allvecs.append(x)
            info['allvecs'] = allvecs

        return x, hits_boundary, info

    # Set default tolerance
    if tol is None:
        tol = max(0.01 * np.sqrt(rt_g), 1e-20)

    # Set maximum iteractions
    if max_iter is None:
        max_iter = n-m
    max_iter = min(max_iter, n-m)

    hits_boundary = False
    stop_cond = 1

    k = 0
    for i in range(max_iter):

        # Stop criteria - Tolerance : r.T g < tol
        if rt_g < tol:
            stop_cond = 4
            break

        k += 1

        # Compute curvature
        pt_H_p = H_p.dot(p)

        # Stop criteria - Negative curvature
        if pt_H_p <= 0:
            if np.isinf(trust_radius):
                    raise ValueError("Negative curvature not "
                                     "allowed for unrestrited "
                                     "problems.")
            else:
                # Find positive value of alpha such:
                # ||x + alpha p|| == trust_radius
                _, alpha = get_boundaries_intersections(x, p, trust_radius)

                x = x + alpha*p
                hits_boundary = True
                stop_cond = 3

                # Store ``x`` value
                if return_all:
                    allvecs.append(x)

                break

        # Get next step
        alpha = rt_g / pt_H_p
        x_next = x + alpha*p

        # Stop criteria - Hits boundary
        if np.linalg.norm(x_next) >= trust_radius:
            # Find positive value of alpha such:
            # ||x + alpha p|| == trust_radius
            _, alpha = get_boundaries_intersections(x, p, trust_radius)

            x = x + alpha*p
            hits_boundary = True
            stop_cond = 2

            break

        # Store ``x_next`` value
        if return_all:
            allvecs.append(x_next)

        # Update residual
        r_next = r + alpha*H_p

        # Project residual g+ = Z r+
        g_next = Z.dot(r_next)

        # Compute conjugate direction step d
        rt_g_next = r_next.dot(g_next)
        beta = rt_g_next / rt_g
        p = - g_next + beta*p

        # Prepare for next iteration
        x = x_next
        g = g_next
        r = g_next
        rt_g = r.dot(g)
        H_p = H.dot(p)

    info = {'niter': k, 'stop_cond': stop_cond}
    if return_all:
        allvecs.append(x)
        info['allvecs'] = allvecs

    return x, hits_boundary, info

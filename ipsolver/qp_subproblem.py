"""
Equality-constrained quadratic programming solvers.
"""

from __future__ import division, print_function, absolute_import
from scipy.sparse import (linalg, bmat, csc_matrix, eye, issparse,
                          isspmatrix_csc, isspmatrix_csr)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
from sksparse.cholmod import cholesky_AAt
from math import copysign
import numpy as np

__all__ = [
    'eqp_kktfact',
    'spherical_boundaries_intersections',
    'box_boundaries_intersections',
    'box_sphere_boundaries_intersections',
    'modified_dogleg',
    'orthogonality',
    'projections',
    'projected_cg'
]


# For comparison with the projected CG
def eqp_kktfact(H, c, A, b):
    """Solve equality-constrained quadratic programming (EQP) problem.

    Solve ``min 1/2 x.T H x + x.t c``  subject to ``A x = b``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H : sparse matrix, shape (n, n)
        Hessian matrix of the EQP problem.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    A : sparse matrix
        Jacobian matrix of the EQP problem.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem.
    lagrange_multipliers : ndarray, shape (m,)
        Lagrange multipliers of the KKT problem.
    """

    n, = np.shape(c)  # Number of parameters
    m, = np.shape(b)  # Number of constraints

    # Karush-Kuhn-Tucker matrix of coeficients.
    # Defined as in Nocedal/Wright "Numerical
    # Optimization" p.452 in Eq. (16.4).
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))

    # Vector of coeficients.
    kkt_vec = np.hstack([-c, b])

    # TODO: Use a symmetric indefinite factorization
    #       to solve the system twice as fast (because
    #       of the symmetry).
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)

    x = kkt_sol[:n]
    lagrange_multipliers = -kkt_sol[n:n+m]

    return x, lagrange_multipliers


def spherical_boundaries_intersections(z, d, trust_radius,
                                       line_intersections=False):
    """Find the intersection between segment (or line) and spherical constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the ball
    ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    trust_radius : float
        Ball radius.
    line_intersections : bool, optional
        When ``True`` the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball
        ``||x|| <= trust_radius``. When ``False`` returns the intersection
        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the ball for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True`` there is a intersection between the line/segment
        and the sphere. On the other hand, when ``False``, there is no
        intersection.
    """

    if np.isinf(trust_radius):
        if line_intersections:
            ta = -np.inf
            tb = np.inf
        else:
            ta = 0
            tb = 1
        intersect = True
        return ta, tb, intersect

    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius**2
    discriminant = b*b - 4*a*c

    if discriminant < 0:
        intersect = False
        return 0, 0, intersect

    sqrt_discriminant = np.sqrt(discriminant)

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

    ta, tb = sorted([ta, tb])

    if line_intersections:
        intersect = True
    else:
        # Checks to see if intersection happens
        # within vectors lenght.
        if tb < 0 or ta > 1:
            intersect = False
            ta = 0
            tb = 0
        else:
            intersect = True

            # Restrict intersection interval
            # between 0 and 1
            ta = max(0, ta)
            tb = min(1, tb)

    return ta, tb, intersect


def box_boundaries_intersections(z, d, lb, ub,
                                 line_intersections=False):
    """Find the intersection between segment (or line) and box constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the retangular box
    ``lb <= x <= ub``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the retangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the retangular box.
    line_intersections : bool, optional
        When ``True`` the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the retangular box.
        When ``False`` returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1``, and the retangular box.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the box for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True`` there is a intersection between the line (or segment)
        and the retangular box. On the other hand, when ``False``, there is no
        intersection.
    """

    # Make sure it is a numpy array
    z = np.asarray(z)
    d = np.asarray(d)
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Get values for which d==0
    zero_d = (d == 0)

    # If the boundaries are not satisfied for some coordinate
    # for which "d" is zero, there is no box-line intersection
    if (z[zero_d] < lb[zero_d]).any() or (z[zero_d] > ub[zero_d]).any():
        intersect = False
        return 0, 0, intersect

    # Remove values for which d is zero
    not_zero_d = np.logical_not(zero_d)
    z = z[not_zero_d]
    d = d[not_zero_d]
    lb = lb[not_zero_d]
    ub = ub[not_zero_d]

    # Find a series of intervals (t_lb[i], t_ub[i])
    t_lb = (lb-z) / d
    t_ub = (ub-z) / d

    # Get the intersection of all those intervals
    ta = max(np.minimum(t_lb, t_ub))
    tb = min(np.maximum(t_lb, t_ub))

    # Check if intersection is feasible
    if ta <= tb:
        intersect = True
    else:
        intersect = False

    # Checks to see if intersection happens
    # within vectors lenght.
    if not line_intersections:
        if tb < 0 or ta > 1:
            intersect = False
            ta = 0
            tb = 0
        else:
            # Restrict intersection interval
            # between 0 and 1
            ta = max(0, ta)
            tb = min(1, tb)

    return ta, tb, intersect


def box_sphere_boundaries_intersections(z, d, lb, ub, trust_radius,
                                        line_intersections=False,
                                        extra_info=False):
    """Find the intersection between segment (or line) and box/sphere constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d``,  the retangular box
    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the retangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the retangular box.
    trust_radius : float
        Ball radius.
    line_intersections : bool, optional
        When ``True`` the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.
        When ``False`` returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.
    extra_info : bool, optional
        When ``True`` returns ``intersect_sphere`` and ``intersect_box``.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the retangular box and
        inside the ball for for ``ta <= t <= tb``.
    intersect : bool
        When ``True`` there is a intersection between the line (or segment)
        and both constraints. On the other hand, when ``False``, there is no
        intersection.
    sphere_info : dictionary, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercept the ball. And a boolean value expliciting
        if it intercepts it or not.
    box_info : Dictionary, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercept the box. And a boolean value expliciting
        if it intercepts it or not.
    """

    ta_b, tb_b, intersect_b = box_boundaries_intersections(z, d, lb, ub,
                                                           line_intersections)
    ta_s, tb_s, intersect_s = spherical_boundaries_intersections(z, d,
                                                                 trust_radius,
                                                                 line_intersections)

    ta = np.maximum(ta_b, ta_s)
    tb = np.minimum(tb_b, tb_s)

    if intersect_b and intersect_s and ta <= tb:
        intersect = True
    else:
        intersect = False

    if extra_info:
        sphere_info = {'ta': ta_s, 'tb': tb_s, 'intersect': intersect_s}
        box_info = {'ta': ta_b, 'tb': tb_b, 'intersect': intersect_b}
        return ta, tb, intersect, sphere_info, box_info
    else:
        return ta, tb, intersect


def inside_box_boundaries(x, lb, ub):
    "Check if lb <= x <= ub."

    return (lb <= x).all() and (x <= ub).all()


def modified_dogleg(A, Y, b, trust_radius, lb, ub):
    """Approximatelly  minimize ``1/2*|| A x + b ||^2`` inside trust-region.

    Approximatelly solve the problem of minimizing ``1/2*|| A x + b ||^2``
    subject to ``||x|| < Delta`` and ``lb <= x <= ub`` using a modification
    of the classical dogleg approach.

    Parameters
    ----------
    A : LinearOperator (or sparse matrix or ndarray), shape (m, n)
        Matrix ``A`` in the minimization problem. It should have
        dimension ``(m, n)`` such that ``m < n``.
    Y : LinearOperator (or sparse matrix or ndarray), shape (n, m)
        LinearOperator that apply the projection matrix
        ``Q = A.T inv(A A.T)`` to the vector.  The obtained vector
        ``y = Q x`` being the minimum norm solution of ``A y = x``.
    b : array_like, shape (m,)
        Vector ``b``in the minimization problem.
    trust_radius: float
        Trust radius to be considered. Delimits a sphere boundary
        to the problem.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``.
        It is expected that ``lb <= 0``, otherwise the algorithm
        may fail. If ``lb[i] = -Inf`` the lower
        bound for the i-th component is just ignored.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``.
        It is expected that ``ub >= 0``, otherwise the algorithm
        may fail. If ``ub[i] = Inf`` the upper bound for the i-th
        component is just ignored.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the problem

    Notes
    -----
    Based on implementations described in p.p. 885-886 from [1]_.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    """

    # Compute minimum norm minimizer of 1/2*|| A x + b ||^2
    newton_point = -Y.dot(b)

    # Check for interior poinr
    if inside_box_boundaries(newton_point, lb, ub)  \
       and np.linalg.norm(newton_point) <= trust_radius:
        x = newton_point
        return x

    # Compute gradient vector ``g = A.T b``
    g = A.T.dot(b)

    # Compute cauchy point
    # `cauchy_point = g.T g / (g.T A.T A g)``
    A_g = A.dot(g)
    cauchy_point = -np.dot(g, g) / np.dot(A_g, A_g) * g

    # Origin
    origin_point = np.zeros_like(cauchy_point)

    # Check the segment between cauchy_point and newton_point
    # for a possible solution
    z = cauchy_point
    p = newton_point - cauchy_point
    _, alpha, intersect = box_sphere_boundaries_intersections(z, p, lb, ub,
                                                              trust_radius)

    if intersect:
        x1 = z + alpha*p

    else:
        # Check the segment between the origin and cauchy_point
        # for a possible solution
        z = origin_point
        p = cauchy_point
        _, alpha, _ = box_sphere_boundaries_intersections(z, p, lb, ub,
                                                          trust_radius)

        x1 = z + alpha*p

    # Check the segment between origin and newton_point
    # for a possible solution
    z = origin_point
    p = newton_point
    _, alpha, _ = box_sphere_boundaries_intersections(z, p, lb, ub,
                                                      trust_radius)

    x2 = z + alpha*p

    # Return the best solution among x1 and x2
    if np.linalg.norm(A.dot(x1) + b) < np.linalg.norm(A.dot(x2) + b):
        return x1
    else:
        return x2


def orthogonality(A, g):
    """Measure of the orthogonality between a vector and the null space of matrix.

    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given vector ``g``.

    The formula is a simplified (and cheaper) version of formula (3.13)
    from [1]_.
    ``orth =  norm(A g, ord=2)/(norm(A, ord='fro')*norm(g, ord=2))``.

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


def projections(A, method=None, orth_tol=1e-12, max_refin=3):
    """Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : sparse matrix (or ndarray), shape (m, n)
        Matrix ``A`` used in the projection.
    method : string, optional
        Method used for compute the given linear
        operators. Should be one of:

            - 'NormalEquation': The operators
               will be computed using the
               so-called normal equation approach
               explained in [1]_. In order to do
               so the Cholesky factorization of
               ``(A A.T)`` is computed. Exclusive
               for sparse matrices.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_. Exclusive
               for sparse matrices.
            - 'QRFactorization': Compute projections
               using QR factorization. Exclusive for
               dense matrices.

    orth_tol : float, optional
        Tolerance for iterative refinements.
    max_refin : int, optional
        Maximum number of iterative refinements

    Returns
    -------
    Z : LinearOperator, shape (n, n)
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator, shape (m, n)
        Least-Square operator. For a given vector ``x``,
        the least-square operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator, shape (n, m)
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``.

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

    # Check Argument
    if issparse(A):
        if method is None:
            method = "AugmentedSystem"

        if method not in ("NormalEquation", "AugmentedSystem"):
            raise ValueError("Method not allowed for the given matrix.")

    else:
        if method is None:
            method = "QRFactorization"

        if method not in ("QRFactorization"):
            raise ValueError("Method not allowed for the given matrix.")

    if method == 'NormalEquation':

        # Cholesky factorization
        factor = cholesky_AAt(A)

        # z = x - A.T inv(A A.T) A x
        def null_space(x):

            v = factor(A.dot(x))
            z = x - A.T.dot(v)

            # Iterative refinement to improve roundoff
            # errors described in [2]_, algorithm 5.1.
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
        #       of the symmetry).
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
            # errors described in [2]_, algorithm 5.2.
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
                lu_sol += lu_update

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

    elif method == "QRFactorization":

        # QRFactorization
        Q, R, P = scipy.linalg.qr(A.T, pivoting=True,  mode='economic')

        # z = x - A.T inv(A A.T) A x
        def null_space(x):

            # v = P inv(R) Q.T x
            aux1 = (Q.T).dot(x)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            v = aux2[P]
            z = x - A.T.dot(v)

            # Iterative refinement to improve roundoff
            # errors described in [2]_, algorithm 5.1.
            k = 0
            while orthogonality(A, z) > orth_tol:
                if k >= max_refin:
                    break

                # v = P inv(R) Q.T x
                aux1 = (Q.T).dot(z)
                aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
                v = aux2[P]

                # z_next = z - A.T v
                z = z - A.T.dot(v)
                k += 1

            return z

        # z = inv(A A.T) A x
        def least_squares(x):

            # z = P inv(R) Q.T x
            aux1 = (Q.T).dot(x)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            z = aux2[P]

            return z

        # z = A.T inv(A A.T) x
        def row_space(x):

            # z = Q inv(R.T) P.T x
            aux1 = np.zeros(m)
            aux1[P] = x
            aux2 = scipy.linalg.solve_triangular(R, aux1,
                                                 lower=False,
                                                 trans='T')
            z = Q.dot(aux2)

            return z

    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)

    return Z, LS, Y


def projected_cg(H, c, Z, Y, b, trust_radius=np.inf,
                 lb=None, ub=None, tol=None,
                 max_iter=None, max_infeasible_iter=None,
                 return_all=False):
    """Solve EQP problem with projected CG method.

    Solve equality-constrained quadratic programming problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x = b`` and,
    possibly, to trust region constraints ``||x|| < trust_radius``
    and box constraints ``lb <= x <= ub``.

    Parameters
    ----------
    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for computing ``H v``.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)
        Operator that, for a given a vector ``b``, compute smallest
        norm solution of ``A x = b``.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.
    trust_radius : float, optional
        Trust radius to be considered. By default uses ``trust_radius=inf``,
        which means no trust radius at all.
    lb : array_like, shape (n,), optional
        Lower bounds to each one of the components of ``x``.
        If ``lb[i] = -Inf`` the lower bound for the i-th
        component is just ignored (default).
    ub : array_like, shape (n, ), optional
        Upper bounds to each one of the components of ``x``.
        If ``ub[i] = Inf`` the upper bound for the i-th
        component is just ignored (default).
    tol : float, optional
        Tolerance used to interrupt the algorithm.
    max_inter : int, optional
        Maximum algorithm iteractions. Where ``max_inter <= n-m``.
        By default uses ``max_iter = n-m``.
    max_infeasible_iter : int, optional
        Maximum infeasible (regarding box constraints) iterations the
        algorithm is allowed to take.
        By default uses ``max_infeasible_iter = n-m``.
    return_all : bool, optional
        When ``true`` return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iteractions.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Reached trust-region boundarie;
                3. Negative curvature detected;
                4. Tolerance was satisfied.
            - allvecs : List containing all intermediary vectors (optional).

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    In the abscence of sperical and box constraints, for sufficient
    iteractions, the method returns a truly optimal result.
    In the presence of those constraints the value returned is only
    a inexpensive approximation of the optimal value.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """

    n, = np.shape(c)  # Number of parameters
    m, = np.shape(b)  # Number of constraints

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

    # If x > trust-region the problem does not have a solution.
    tr_distance = trust_radius - np.linalg.norm(x)
    if tr_distance < 0:
        raise ValueError("Trust region problem does not have a solution.")

    # If x == trust_radius, then x is the solution
    # to the optimization problem, since x is the
    # minimum norm solution to Ax=b.
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

    # Set default lower and upper bounds
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)

    # Set maximum iteractions
    if max_iter is None:
        max_iter = n-m
    max_iter = min(max_iter, n-m)

    # Set maximum infeasible iteractions
    if max_infeasible_iter is None:
        max_infeasible_iter = n-m

    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.empty_like(x)

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
                # Find intersection with constraints
                _, alpha, intersect = box_sphere_boundaries_intersections(x, p, lb, ub,
                                                                          trust_radius,
                                                                          line_intersections=True)

                if intersect:
                    x = x + alpha*p
                stop_cond = 3
                hits_boundary = True
                break

        # Get next step
        alpha = rt_g / pt_H_p
        x_next = x + alpha*p

        # Stop criteria - Hits boundary
        if np.linalg.norm(x_next) >= trust_radius:
            # Find intersection with box constraints
            _, theta, intersect = box_sphere_boundaries_intersections(x, alpha*p, lb, ub,
                                                                      trust_radius)

            if intersect:
                x = x + theta*alpha*p
            stop_cond = 2
            hits_boundary = True
            break

        # Check if ``x`` is inside box contraints
        # and start counter if it is not
        if inside_box_boundaries(x_next, lb, ub):
            counter = 0
        else:
            counter += 1

        # Whenever outside box constraints keep looking for intersections
        if counter > 0:
            _, theta, intersect = box_sphere_boundaries_intersections(x, alpha*p, lb, ub,
                                                                      trust_radius)

            if intersect:
                last_feasible_x = x + theta*alpha*p
                counter = 0

        # Stop after too many infeasible (regarding box constraints) iteration
        if counter > max_infeasible_iter:
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

    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True

    info = {'niter': k, 'stop_cond': stop_cond}
    if return_all:
        info['allvecs'] = allvecs

    return x, hits_boundary, info

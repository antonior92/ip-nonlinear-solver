"""
Implement Byrd-Omojokun Trust-Region SQP method
"""
from __future__ import division, print_function, absolute_import
import scipy.sparse as spc
from .projections import projections
from .qp_subproblem import qp_subproblem, inside_box_boundaries
import numpy as np
from numpy.linalg import norm

__all__ = [
    'equality_constrained_sqp',
]


def default_scalling(x):
    n, = np.shape(x)
    return spc.eye(n)


def default_stop_criteria(x, v, fun, grad, constr, jac, iteration,
                          trust_radius):
    opt = norm(grad + jac.T.dot(v))
    c_violation = norm(constr)

    if (opt < 1e-8 and c_violation < 1e-8) or iteration > 1000 \
       or trust_radius < 1e-12:
        return True
    else:
        return False


def equality_constrained_sqp(fun, grad, hess, constr, jac,
                             x0, v0=None,
                             initial_trust_radius=1.0,
                             trust_lb=None,
                             trust_ub=None,
                             stop_criteria=default_stop_criteria,
                             initial_penalty=1.0,
                             scaling=default_scalling,
                             return_all=False):
    """Solve nonlinear equality-constrained problem using trust-region SQP.

    Solve problem:

        minimize f(x)
        subject to: c(x) = 0

    using Byrd-Omojokun Trust-Region SQP method.

    Parameters
    ----------
    fun : callable
        Objective function:
            fun(x) -> float
    grad : callable
        Gradient vector:
            grad(x) -> array_like, shape (n,)
    hess : callable
        Lagrangian hessian:
            hess(x, v) -> LinearOperator (or sparse matrix or ndarray), shape (n, n)
    constr : callable
        Equality constraint:
            constr(x) -> array_like, shape (m,)
    jac : callable
        Constraints Jacobian:
            jac(x) -> sparse matrix (or ndarray), shape (m, n)
    x0 : array_like, shape (n,)
        Starting point.
    v0 : array_like, shape (n,)
        Initial lagrange multipliers. By default uses least-squares lagrange
        multipliers.
    initial_trust_radius: float
        Initial trust-region radius. By defaut uses 1.
    trust_lb : array_like, shape (n,), optional
        Trust region lower bound.
    trust_ub : array_like, shape (n,), optional
        Trust region upper bound.
    stop_criteria: callable
        Functions that returns True when stop criteria is fulfilled:
            stop_criteria(x, v, fun, grad, constr, jac,
                          iteration, trust_radius)
    initial_penalty : float
        Initial penalty for merit function.
    scaling : callable
        Function that return scaling used by the trust region:
            scaling(x) -> LinearOperator (or sparse matrix or ndarray), shape (n, n)
    return_all : bool, optional
        When ``true`` return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the equality constrained problem.
    info :
        Dictionary containing the following:

            - niter : Number of iterations.
            - trust_radius : Trust radius at last iteration.
            - v : Lagrange multipliers at the solution , shape (m,).
            - fun : Function evaluation at the solution.
            - grad : Gradient evaluation at the solution.
            - hess : Lagrangian Hessian at the solution.
            - constr : Constraints at the solution.
            - jac : Constraints jacobian at the solution.
            - opt : Optimality is the norm of gradient of the Lagrangian
              ``||grad L(x, v)||``, where ``grad L(x, v) = g(x) + A(x).T v``.
            - c_violation : Norm of the constraint violation ``||c(x)||``.
            - allvecs : List containing all intermediary vectors (optional).
            - allmult : List containing all intermediary lagrange
                        multipliers (optional).

    Notes
    -----
    This algorithm is a variation of Byrd-Omojokun Trust-Region
    SQP method (described in [2]_ p.549). The details of this specific
    implementation are inspired by [1]_ and it is used as a substep
    of a interior point method.

    At each substep solve, using the projected CG method, the trust-region
    QP subproblem:

        minimize c.T d + 1/2 d.T H d
        subject to : A d + b = 0
                    ||d|| <= trust_radius
                    trust_lb <= d <= trust_ub

    and update the solution ``x += d``.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [2] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """
    PENALTY_FACTOR = 0.3  # Rho from formula (3.51), reference [1]_, p.891.
    LARGE_REDUCTION_RATIO = 0.9
    INTERMEDIARY_REDUCTION_RATIO = 0.3
    SUFFICIENT_REDUCTION_RATIO = 1e-8  # Eta described on reference [1]_, p.892.
    TRUST_REDUCTION_FACTOR = 0.5
    TRUST_ENLARGEMENT_FACTOR_L = 7
    TRUST_ENLARGEMENT_FACTOR_S = 2
    MAX_TRUST_REDUCTION = 0.5
    MIN_TRUST_REDUCTION = 0.1
    SOC_THRESHOLD = 0.1

    n, = np.shape(x0)  # Number of parameters

    # Set default lower and upper bounds.
    if trust_lb is None:
        trust_lb = np.full(n, -np.inf)
    if trust_ub is None:
        trust_ub = np.full(n, np.inf)

    # Initial values
    iteration = 0
    x = np.copy(x0)
    trust_radius = initial_trust_radius
    penalty = initial_penalty
    # Compute Values
    f = fun(x)
    c = grad(x)
    b = constr(x)
    A = jac(x)
    # Get projections
    Z, LS, Y = projections(A)
    # Set initial lagrange multipliers
    if v0 is None:
        # Compute least-square lagrange multipliers
        v = -LS.dot(c)
    else:
        v = v0
    # Store values
    if return_all:
        allvecs = [np.copy(x)]
        allmult = [np.copy(v)]

    compute_hess = True
    while not stop_criteria(x, v, f, c, b, A, iteration, trust_radius):
        # Compute Lagrangian Hessian
        if compute_hess:
            H = hess(x, v)

        # Solve QP subproblem
        dn, dt, info_cg = qp_subproblem(H, c, A, Z, Y, b,
                                        trust_radius,
                                        trust_lb, trust_ub)
        # Compute update (normal + tangential steps).
        d = dn + dt

        # Compute second order model: 1/2 d H d + g.T d + f.
        quadratic_model = 1/2*(H.dot(d)).dot(d) + c.T.dot(d)
        # Compute linearized constraint: l = A d + c.
        linearized_constr = A.dot(d)+b
        # Compute new penalty parameter according to formula (3.52),
        # reference [1]_, p.891.
        vpred = norm(b) - norm(linearized_constr)
        if vpred != 0:
            new_penalty = quadratic_model / ((1-PENALTY_FACTOR)*vpred)
            penalty = max(penalty, new_penalty)
        # Compute predicted reduction according to formula (3.52),
        # reference [1]_, p.891.
        predicted_reduction = -quadratic_model + penalty*vpred

        # Compute merit function at current point
        merit_function = f + penalty*norm(b)
        # Evaluate function and constraints at trial point
        f_next = fun(x+d)
        b_next = constr(x+d)
        # Compute merit function at trial point
        merit_function_next = f_next + penalty*norm(b_next)
        # Compute actual reduction according to formula (3.54),
        # reference [1]_, p.892.
        actual_reduction = merit_function - merit_function_next
        # Compute reduction ratio
        reduction_ratio = actual_reduction / predicted_reduction

        # Reajust trust region step, formula (3.55), reference [1]_, p.892.
        if reduction_ratio >= LARGE_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * norm(d),
                               trust_radius)
        elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * norm(d),
                               trust_radius)
        elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
            # Second order correction (SOC), reference [1]_, p.892.
            if norm(dn) <= SOC_THRESHOLD * norm(dt):
                # Update steps
                y = Y.dot(b_next)
                # Recompute ared
                f_soc = fun(x+d+y)
                b_soc = constr(x+d+y)
                merit_function_soc = f_soc + penalty*norm(b_soc)
                actual_reduction_soc = merit_function - merit_function_soc
                # Recompute reduction ratio
                reduction_ratio_soc = actual_reduction_soc / predicted_reduction
                if reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO and \
                   inside_box_boundaries(d+y, trust_lb, trust_ub):
                    d = d + y
                    f_next = f_soc
                    b_next = b_soc
                    reduction_ratio = reduction_ratio_soc
                else:
                    new_trust_radius = TRUST_REDUCTION_FACTOR * norm(d)
                    if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                        trust_radius *= MAX_TRUST_REDUCTION
                    elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                        trust_radius = new_trust_radius
                    else:
                        trust_radius *= MIN_TRUST_REDUCTION
            else:
                new_trust_radius = TRUST_REDUCTION_FACTOR * norm(d)
                if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                    trust_radius *= MAX_TRUST_REDUCTION
                elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                    trust_radius = new_trust_radius
                else:
                    trust_radius *= MIN_TRUST_REDUCTION

        # Update iteration
        iteration += 1
        if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
            S = scaling(x)
            x += S.dot(d)
            f = f_next
            c = grad(x)
            b = b_next
            A = jac(x)
            # Get projections
            Z, LS, Y = projections(A)
            # Compute least-square lagrange multipliers
            v = -LS.dot(c)
            # Set Flag
            compute_hess = True
        else:
            compute_hess = False
        # Store values
        if return_all:
            allvecs.append(np.copy(x))
            allmult.append(np.copy(v))

    opt = norm(c + A.T.dot(v))
    constr = norm(b)
    info = {'niter': iteration, 'trust_radius': trust_radius,
            'v': v, 'fun': f,
            'grad': c, 'hess': H,
            'constr': b, 'jac': A,
            'opt': opt,
            'c_violation': constr}
    if return_all:
        info['allvecs'] = allvecs
        info['allmult'] = allmult

    return x, info

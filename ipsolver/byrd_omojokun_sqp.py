""" 
Implement Byrd-Omojokun Trust-Region SQP method
"""
from __future__ import division, print_function, absolute_import
from scipy.sparse import (issparse)
from scipy.sparse.linalg import LinearOperator
from .projections import projections
from .qp_subproblem import qp_subproblem
import numpy as np
from numpy.linalg import norm


def byrd_omojokun_sqp(f, g, H, c, A, x0, trust_radius0,
                      stop_criteria, tr_scaling,
                      tr_lb=None, tr_ub=None,
                      penalty0=1):
    """Solve nonlinear equality-constrained problem using trust-region SQP.

    Solve problem:

        minimize f(x)
        subject to: c(x) = 0

    using Byrd-Omojokun Trust-Region SQP method. Solve, at each substep, the
    trust-region equality-constrained QP problems using projected CG.

    Parameters
    ----------
    f : callable
        Objective function:
            f(x) -> float
    g : callable
        Gradient vector:
            g(x) -> array_like, shape (n,)
    H : callable
        Lagrangian hessian:
            H(x, lambda) -> LinearOperator (or sparse matrix or ndarray), shape (n, n)
    c : callable
        Equality constraint:
            c(x) -> array_like, shape (m,)
    A : callable
        Constraints Jacobian:
            A(x) -> sparse matrix (or ndarray), shape (m, n)
    x_0 : array_like, shape (n,)
        Starting point.
    trust_radius0: float
        Initial trust-region
    stop_criteria: callable
        Functions that returns True when stop criteria is fulfilled:
            stop_criteria(xk, lambdak, fk, gk, ck, Ak, k, trust_radiusk)
    tr_scaling : callable
        Function that return scaling for the trust region:
            tr_scaling(x) -> array_like, shape (n,)
    tr_lb : array_like, shape (n,), optional
        Lower bound for step (after the scaling).
    tr_ub : array_like, shape (n,), optional
        Upper bound for step (after the scaling).
    penalty0 : float
        Initial penalty for merit function.

    Notes
    -----
    This algorithm is a variation of Byrd-Omojokun Trust-Region
    SQP method (described in [3] p.549).

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [2] Byrd, Richard H., Jean Charles Gilbert, and Jorge Nocedal.
           "A trust region method based on interior point techniques
           for nonlinear programming." Mathematical Programming 89.1
           (2000): 149-185.
    .. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """
    PENALTY_FACTOR = 0.3  # Rho from formula (3.51), reference [1]_, p.891.
    SUFFICIENT_REDUCTION = 1e-8  # Eta described on reference [1]_, p.892.

    n, = np.shape(x0)  # Number of parameters

    # Set default lower and upper bounds.
    if tr_lb is None:
        tr_lb = np.full(n, -np.inf)
    if tr_ub is None:
        tr_ub = np.full(n, np.inf)

    # Initial values
    k = 0
    xk = x0
    trust_radiusk = trust_radius0
    penaltyk = penalty0
    # Compute Values
    fk = f(xk)
    gk = g(xk)
    ck = c(xk)
    Ak = A(xk)
    norm_ck = norm(ck)
    # Apply scaling
    Sk = tr_scaling(xk)
    gkSk = gk*Sk
    if issparse(Ak):
        AkSk = Ak.multiply(Sk)
    else:
        AkSk = np.multiply(Ak, Sk)
    # Get projections
    Zk, LSk, Yk = projections(AkSk)
    # Compute least-square lagrange multipliers
    lambda_k = -LSk.dot(gk)
    compute_Hk = True
    while not stop_criteria(xk, lambda_k, fk, gk, ck, Ak, k, trust_radiusk):
        # Compute Lagrangian Hessian
        if compute_Hk:
            Hk = H(xk, lambda_k)

            # Apply scaling
            def matvec(x):
                return Sk*Hk.dot(Sk*x)
            SkHkSk = LinearOperator((n, n), matvec=matvec)
        # Solve QP subproblem
        dn, dt, r, hits_boundary, info_cg = qp_subproblem(SkHkSk, gkSk, AkSk,
                                                          Zk, Yk, ck,
                                                          trust_radiusk,
                                                          tr_lb, tr_ub)
        # Computer update (normal + tangential steps).
        dk = dn + dt

        # Compute second order model: q = 1/2 d H d + g.T d + f.
        qk = 1/2*(SkHkSk.dot(dk)).dot(dk) + gkSk.T.dot(dk)
        # Compute linearized constraint: l = A d + c.
        lk = AkSk.dot(dk)+ck
        # Compute new penalty parameter according to formula (3.52),
        # reference [1]_, p.891.
        vpred = norm_ck - norm(lk)
        if vpred != 0:
            new_penalty = qk/((1-PENALTY_FACTOR)*vpred)
            penaltyk = max(penaltyk, new_penalty)
        # Compute predicted reduction according to formula (3.52),
        # reference [1]_, p.891.
        pred = -qk + penaltyk*vpred

        # Compute merit function at current point
        merit_function_xk = fk + penaltyk*norm_ck
        # Evaluate function and constraints at trial point
        fk_next = f(xk+dk)
        ck_next = c(xk+dk)
        norm_ck_next = norm(ck_next)
        # Compute merit function at trial point
        merit_function_xk_next = fk_next + penaltyk*norm_ck_next
        # Compute actual reduction according to formula (3.54),
        # reference [1]_, p.892.
        ared = merit_function_xk - merit_function_xk_next
        # Compute reduction ratio
        reduction_ratio = ared/pred

        # Second order correction, reference [1]_, p.892.
        if reduction_ratio < SUFFICIENT_REDUCTION and norm(dn) <= 0.1*norm(dt):
            # Update steps
            y = LSk.dot(ck_next)
            dk = dk + y
            # Recompute pred
            qk = 1/2*(SkHkSk.dot(dk)).dot(dk) + gkSk.T.dot(dk)
            lk = AkSk.dot(dk)+ck
            pred = -qk + penaltyk*(norm_ck - norm(lk))
            # Recompute ared
            ck_next = c(xk+dk)
            norm_ck_next = norm(ck_next)
            merit_function_xk_next = fk_next + penaltyk*norm_ck_next
            ared = merit_function_xk - merit_function_xk_next
            # Recompute reduction ratio
            reduction_ratio = ared/pred

        # Reajust trust region step, formula (3.55), reference [1]_, p.892.
        if reduction_ratio >= 0.9:
            trust_radiusk = max(7*norm(dk), trust_radiusk)
        elif reduction_ratio >= 0.3:
            trust_radiusk = max(2*norm(dk), trust_radiusk)
        elif reduction_ratio >= SUFFICIENT_REDUCTION:
            trust_radiusk = trust_radiusk
        else:
            new_trust_radius = 0.5*norm(dk)
            if new_trust_radius >= 1/2*trust_radiusk:
                trust_radiusk = 1/2*trust_radiusk
            elif new_trust_radius >= 1/10*trust_radiusk:
                trust_radiusk = new_trust_radius
            else:
                trust_radiusk = 1/10*trust_radiusk

        # Update iteration
        k += 1
        if reduction_ratio >= SUFFICIENT_REDUCTION:
            fk = f(xk)
            gk = g(xk)
            ck = c(xk)
            Ak = A(xk)
            norm_ck = norm(ck)
            # Apply scaling
            Sk = tr_scaling(xk)
            gkSk = gk*Sk
            if issparse(Ak):
                AkSk = Ak.multiply(Sk)
            else:
                AkSk = np.multiply(Ak, Sk)
            # Get projections
            Zk, LSk, Yk = projections(AkSk)
            # Compute least-square lagrange multipliers
            lambda_k = -LSk.dot(gk)
            # Set Flag
            compute_Hk = True
        else:
            compute_Hk = False

    return xk, lambda_k, trust_radiusk

"""
Implement Byrd-Omojokun Trust-Region SQP method
"""
from __future__ import division, print_function, absolute_import
from scipy.sparse import (issparse)
from scipy.sparse.linalg import LinearOperator
from .projections import projections
from .qp_subproblem import qp_subproblem, inside_box_boundaries
import numpy as np
from numpy.linalg import norm
import pdb

__all__ = [
    'byrd_omojokun_sqp',
]


def default_tr_scalling(x):
    return np.ones_like(x)


def default_stop_criteria(xk, vk, fk, gkSk, ck, AkSk, k, trust_radiusk):
    opt = norm(gkSk + AkSk.T.dot(vk))
    const = norm(ck)

    if (opt < 1e-8 and const < 1e-8) or k > 1000 or trust_radiusk < 1e-12:
        return True
    else:
        return False


def byrd_omojokun_sqp(f, g, H, c, A, x0,
                      v0=None,
                      stop_criteria=default_stop_criteria,
                      tr_scaling=default_tr_scalling,
                      trust_radius0=1.0,
                      penalty0=1.0,
                      tr_lb=None, tr_ub=None,
                      return_all=False):
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
            H(x, v) -> LinearOperator (or sparse matrix or ndarray), shape (n, n)
    c : callable
        Equality constraint:
            c(x) -> array_like, shape (m,)
    A : callable
        Constraints Jacobian:
            A(x) -> sparse matrix (or ndarray), shape (m, n)
    x_0 : array_like, shape (n,)
        Starting point.
    v_0 : array_like, shape (n,)
        Initial lagrange multipliers. By default uses least-squares lagrange
        multipliers.
    trust_radius0: float
        Initial trust-region
    stop_criteria: callable
        Functions that returns True when stop criteria is fulfilled:
            stop_criteria(xk, vk, fk, gkSk, ck, AkSk, k, trust_radiusk)
    tr_scaling : callable
        Function that return scaling for the trust region:
            tr_scaling(x) -> array_like, shape (n,)
    penalty0 : float
        Initial penalty for merit function.
    tr_lb : array_like, shape (n,), optional
        Lower bound for step (after the scaling).
    tr_ub : array_like, shape (n,), optional
        Upper bound for step (after the scaling).
    return_all : bool, optional
        When ``true`` return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the equality constrained problem.
    info :
        Dictionary containing the following:

            - niter : Number of iteractions.
            - trust_radius : Trust radius at last iteraction.
            - v : Lagrange multipliers at the solution , shape (m,).
            - f : Function evaluation at the solution.
            - g : Gradient evaluation at the solution.
            - H : Lagrangian Hessian at the solution.
            - c : Constraints at the solution.
            - A : Constraints jacobian at the solution.
            - opt : Optimality is the norm of gradient of the Lagrangian
                    ``||grad L(x, v)||``,
                     where ``grad L(x, v) = g(x) + A(x).T v``.
            - constr : Norm of the constraint violation ``||c(x)||``
            - allvecs : List containing all intermediary vectors (optional).
            - allmult : List containing all intermediary lagrange
                        multipliers (optional).

    Notes
    -----
    This algorithm is a variation of Byrd-Omojokun Trust-Region
    SQP method (described in [2]_ p.549). The details of this specific
    implementation are inspired by [1]_ and is used as a substep
    of a larger solver.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [2] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
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
    xk = np.copy(x0)
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
    # Set initial lagrange multipliers
    if v0 is None:
        # Compute least-square lagrange multipliers
        vk = -LSk.dot(gkSk)
    else:
        vk = v0
    # Store values
    if return_all:
        allvecs = [np.copy(xk)]
        allmult = [np.copy(vk)]

    compute_Hk = True
    while not stop_criteria(xk, vk, fk, gkSk, ck, AkSk, k, trust_radiusk):
        # Compute Lagrangian Hessian
        if compute_Hk:
            Hk = H(xk, vk)

            # Apply scaling
            def matvec(x):
                return Sk*Hk.dot(Sk*x)
            SkHkSk = LinearOperator((n, n), matvec=matvec)
        # Solve QP subproblem
        dn, dt, r, hits_boundary, info_cg = qp_subproblem(SkHkSk, gkSk, AkSk,
                                                          Zk, Yk, ck,
                                                          trust_radiusk,
                                                          tr_lb, tr_ub)
        # Compute update (normal + tangential steps).
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

        # Reajust trust region step, formula (3.55), reference [1]_, p.892.
        if reduction_ratio >= 0.9:
            trust_radiusk = max(7*norm(dk), trust_radiusk)
        elif reduction_ratio >= 0.3:
            trust_radiusk = max(2*norm(dk), trust_radiusk)
        elif reduction_ratio < SUFFICIENT_REDUCTION:
            # Second order correction, reference [1]_, p.892.
            if norm(dn) <= 0.1*norm(dt):
                # Update steps
                y = Yk.dot(ck_next)
                # Recompute ared
                fk_soc = f(xk+dk+y)
                ck_soc = c(xk+dk+y)
                norm_ck_soc = norm(ck_next)
                merit_function_xk_soc = fk_soc + penaltyk*norm_ck_soc
                ared_soc = merit_function_xk - merit_function_xk_soc
                # Recompute reduction ratio
                reduction_ratio_soc = ared_soc/pred
                if reduction_ratio_soc >= SUFFICIENT_REDUCTION and \
                   inside_box_boundaries(dk+y, tr_lb, tr_ub):
                    dk = dk + y
                    fk_next = fk_soc
                    ck_next = ck_soc
                    norm_ck_next = norm_ck_soc
                    reduction_ratio = reduction_ratio_soc
                else:
                    new_trust_radius = 0.5*norm(dk)
                    if new_trust_radius >= 1/2*trust_radiusk:
                        trust_radiusk = 1/2*trust_radiusk
                    elif new_trust_radius >= 1/10*trust_radiusk:
                        trust_radiusk = new_trust_radius
                    else:
                        trust_radiusk = 1/10*trust_radiusk
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
            xk += dk
            fk = fk_next
            gk = g(xk)
            ck = ck_next
            Ak = A(xk)
            norm_ck = norm_ck_next
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
            vk = -LSk.dot(gkSk)
            # Set Flag
            compute_Hk = True
        else:
            compute_Hk = False
        # Store values
        if return_all:
            allvecs.append(np.copy(xk))
            allmult.append(np.copy(vk))

    opt = norm(gk + Ak.T.dot(vk))
    constr = norm(ck)
    info = {'niter': k, 'trust_radius': trust_radiusk, 'v': vk, 'f': fk,
            'g': gk, 'H': Hk, 'c': ck, 'A': Ak, 'opt': opt,
            'constr': constr}
    if return_all:
        info['allvecs'] = allvecs
        info['allmult'] = allmult

    return xk, info

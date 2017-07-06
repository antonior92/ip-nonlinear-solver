"""Trust-region interior points methods"""

from __future__ import division, print_function, absolute_import
import scipy.sparse as spc
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

__all__ = ['BarrierSubproblem',
           'ipsolver']


class BarrierSubproblem:
    """
    Barrier optimization problem:

        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to: constr_eq(x)     = 0
                      A_eq x + b     = 0
                       constr(x) + s = 0
                         A x + b + s = 0
                          x - ub + s = 0  (for ub != inf)
                          lb - x + s = 0  (for lb != -inf)
    """

    def __init__(self, x0, fun, grad, hess, constr, jac,
                 constr_eq, jac_eq, lb, ub, A, b, A_eq, b_eq,
                 barrier_parameter, tolerance, max_substep_iter):
        # Compute number of variables
        self.n_vars, = np.shape(x0)

        # Define empty default functions
        def empty_constr(x):
            return np.empty((0,))

        def empty_jac(x):
            return np.empty((0, self.n_vars))

        # Store parameters
        self.x0 = x0
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.constr = constr if constr is not None else empty_constr
        self.jac = jac if jac is not None else empty_jac
        self.constr_eq = constr_eq if constr_eq is not None else empty_constr
        self.jac_eq = jac_eq if jac_eq is not None else empty_jac
        self.A = A if A is not None else np.empty((0, self.n_vars))
        self.b = b if b is not None else np.empty((0,))
        self.A_eq = A_eq if A_eq is not None else np.empty((0, self.n_vars))
        self.b_eq = b_eq if b_eq is not None else np.empty((0,))
        self.lb = lb if lb is not None else np.full((self.n_vars,), -np.inf)
        self.ub = ub if ub is not None else np.full((self.n_vars,), np.inf)
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.max_substep_iter = max_substep_iter

        # Compute number of finite lower and upper bounds
        self.ind_lb = np.invert(np.isinf(self.lb))
        self.n_lb = np.int(np.sum(self.ind_lb))
        self.ind_ub = np.invert(np.isinf(self.ub))
        self.n_ub = np.int(np.sum(self.ind_ub))
        # Nonlinear constraints
        # TODO: Avoid this unecessary call to the constraints.
        self.n_eq, = np.shape(self.constr_eq(x0))
        self.n_ineq, = np.shape(self.constr(x0))
        # Linear constraints
        self.n_lin_eq, = np.shape(self.b_eq)
        self.n_lin_ineq, = np.shape(self.b)
        # Number of slack variables
        self.n_slack = (self.n_lb + self.n_ub +
                        self.n_ineq + self.n_lin_ineq)

    def update(self, barrier_parameter, tolerance):
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance

    def get_slack(self, z):
        return z[self.n_vars:self.n_vars+self.n_slack]

    def get_variables(self, z):
        return z[:self.n_vars]

    def s0(self):
        return np.ones(self.n_slack)

    def z0(self):
        return np.hstack((self.x0, self.s0()))

    def function(self, z):
        """Returns barrier function at given point.

        For z = [x, s], returns barrier function:
            function(z) = fun(x) - barrier_parameter*sum(log(s))
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        return self.fun(x) - self.barrier_parameter*np.sum(np.log(s))

    def constraints(self, z):
        """Returns barrier problem constraints at given points.

        For z = [x, s], returns the constraints:

            constraints(z) = [[   constr_eq(x)   ]]
                             [[    A_eq x + b    ]]
                             [[[ constr(x) ]     ]]
                             [[[   A x + b ]     ]]
                             [[[    x - ub ] + s ]]  (for ub != inf)
                             [[[    lb - x ]     ]]  (for lb != -inf)
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        aux = np.hstack((self.constr(x),
                         self.A.dot(x) + self.b,
                         x[self.ind_ub] - self.ub[self.ind_ub],
                         self.lb[self.ind_lb] - x[self.ind_lb]))
        return np.hstack((self.constr_eq(x),
                          self.A_eq.dot(x) + self.b_eq,
                          aux + s))

    def scaling(self, z):
        """Returns scaling vector.

        Given by:
            scaling = [ones(n_vars), s]
        """
        s = self.get_slack(z)
        diag_elements = np.hstack((np.ones(self.n_vars), s))

        # Diagonal Matrix
        def matvec(vec):
            return diag_elements*vec
        return LinearOperator((self.n_vars+self.n_slack,
                               self.n_vars+self.n_slack),
                              matvec)

    def gradient(self, z):
        """Returns scaled gradient.

        Barrier  scalled gradient
        of the barrier problem by the previously
        defined scaling factor:
            gradient = [[             grad(x)             ]]
                       [[ -barrier_parameter*ones(n_ineq) ]]
        """
        x = self.get_variables(z)
        return np.hstack((self.grad(x),
                          -self.barrier_parameter*np.ones(self.n_slack)))

    def jacobian(self, z):
        """Returns scaled Jacobian.

        Barrier scalled jacobian
        by the previously defined scaling factor:
            jacobian = [[  jac_eq(x)     0  ]]
                       [[  A_eq(x)       0  ]]
                       [[[ jac(x) ]         ]]
                       [[[   A    ]         ]]
                       [[[   I    ]      S  ]]
                       [[[  -I    ]         ]]
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        S = spc.diags((s,), (0,))
        I = spc.eye(self.n_vars).tocsc()
        I_ub = I[self.ind_ub, :]
        I_lb = I[self.ind_lb, :]

        aux = spc.vstack([self.jac(x),
                          self.A,
                          I_ub,
                          -I_lb])
        return spc.bmat([[self.jac_eq(x), None],
                         [self.A_eq, None],
                         [aux, S]], "csc")

    def lagrangian_hessian_x(self, z, v):
        """Returns Lagrangian Hessian (in relation to variables ``x``)"""
        x = self.get_variables(z)
        # Get lagrange multipliers relatated to nonlinear equality constraints
        v_eq = v[:self.n_eq]
        # Get lagrange multipliers relatated to nonlinear ineq. constraints
        v_ineq = v[self.n_eq+self.n_lin_eq:self.n_eq+self.n_lin_eq+self.n_ineq]
        hess = self.hess
        return hess(x, v_eq, v_ineq)

    def lagrangian_hessian_s(self, z, v):
        """Returns Lagrangian Hessian (in relation to slack variables ``s``)"""
        s = self.get_slack(z)
        # Using the primal formulation:
        #     lagrangian_hessian_s = diag(1/s**2).
        # Reference [1]_ p. 882, formula (3.1)
        primal = self.barrier_parameter/(s*s)
        # Using the primal-dual formulation
        #     lagrangian_hessian_s = diag(v/s)
        # Reference [1]_ p. 883, formula (3.11)
        primal_dual = v[-self.n_slack:]/s
        # Uses the primal-dual formulation for
        # positives values of v_ineq, and primal
        # formulation for the remaining ones.
        return np.where(v[-self.n_slack:] > 0, primal_dual, primal)

    def lagrangian_hessian(self, z, v):
        """Returns scaled Lagrangian Hessian"""
        s = self.get_slack(z)
        # Compute Hessian in relation to x and s
        Hx = self.lagrangian_hessian_x(z, v)
        Hs = self.lagrangian_hessian_s(z, v)*s*s

        # The scaled Lagragian Hessian is:
        #     [[ Hx    0    ]]
        #     [[ 0   S Hs S ]]
        def matvec(vec):
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            return np.hstack((Hx.dot(vec_x), Hs*vec_s))
        return LinearOperator((self.n_vars+self.n_slack,
                               self.n_vars+self.n_slack),
                              matvec)

    def stop_criteria(self, info):
        """Stop criteria to the barrier problem.

        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        if (info["opt"] < self.tolerance
            and info["constr_violation"] < self.tolerance) \
           or info["niter"] > self.max_substep_iter:
            return True
        else:
            return False


def default_stop_criteria(info):
    if (info["opt"] < 1e-8 and info["constr_violation"] < 1e-8) \
       or info["niter"] > 1000:
        return True
    else:
        return False


def ipsolver(fun, grad, hess, x0, constr=None, jac=None,
             constr_eq=None, jac_eq=None, A=None, b=None,
             A_eq=None, b_eq=None, lb=None, ub=None,
             stop_criteria=default_stop_criteria,
             initial_barrier_parameter=0.1,
             initial_tolerance=0.1,
             initial_penalty=1.0,
             initial_trust_radius=1.0,
             max_substep_iter=1000):
    """Trust-region interior points method.

    Solve problem:

        minimize fun(x)
        subject to: constr(x) <= 0
                  constr_eq(x) = 0
                          A x <= b
                        A_eq x = b
                         lb <= x <= ub

    using trust-region interior point method described in [1]_.

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
            hess(x, v_eq, v_ineq) -> H

            - ``x``: array_like, shape (n,)
                Evaluation point.
            - ``v_eq``: array_like, shape (n_eq,)
                Lagrange multipliers for equality constraints.
            - ``v_ineq``: array_like, shape (n_ineq,)
                Lagrange multipliers for inequality constraints.
            - ``H``: LinearOperator (or sparse matrix or ndarray), shape (n, n)
                Lagrangian Hessian.

    constr : callable
        Inequality constraint:
            constr(x) -> array_like, shape (n_ineq,)
    jac : callable
        Inequality constraints Jacobian:
            jac(x) -> sparse matrix (or ndarray), shape (n_ineq, n)
    constr_eq : callable
        Inequality constraint:
            constr(x) -> array_like, shape (n_eq,)
    jac_eq : callable
        Inequality constraints Jacobian:
            jac(x) -> sparse matrix (or ndarray), shape (n_eq, n)
    lb : array_like, shape (n,)
        Lower bound.
    ub : array_like, shape (n,)
        Upper bound.
    A : sparse matrix (or ndarray), shape (n_lin_ineq, n)
        Jacobian of linear inequality constraint.
    b : array_like, shape (n_lin_ineq, n)
        Right-hand side of linear inequality constraint.
    A_eq : sparse matrix (or ndarray), shape (n_lin_eq, n)
        Jacobian of linear equality constraint.
    b_eq : array_like, shape (n_lin_eq, n)
        Right-hand side of linear equality constraint.
    x0 : array_like, shape (n,)
        Starting point.
    stop_criteria: callable
        Functions that returns True when stop criteria is fulfilled:
            stop_criteria(info)
    initial_tolerance: float
        Initial subproblem tolerance. By defaut uses 0.1.
    initial_barrier_parameter: float
        Initial barrier parameter. By defaut uses 0.1.
    initial_trust_radius: float
        Initial trust-region radius. By defaut uses 1.
    initial_penalty : float
        Initial penalty for merit function.
    max_substep_iter : int
        Maximum iterations per substep.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the optimization problem.
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

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
           "On the local behavior of an interior point method for
           nonlinear programming." Numerical analysis 1997 (1997): 37-56.
    """
    # BOUNDARY_PARAMETER controls the decrease on the slack
    # variables. Represents ``tau`` from [1]_ p.885, formula (3.18).
    BOUNDARY_PARAMETER = 0.995
    # BARRIER_DECAY_RATIO controls the decay of the barrier parameter
    # and of the subproblem toloerance. Represents ``theta`` from [1]_ p.879.
    BARRIER_DECAY_RATIO = 0.2
    # TRUST_ENLARGEMENT controls the enlargement on trust radius
    # after each iteration
    TRUST_ENLARGEMENT = 5

    # Initial Values
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    v = None
    iteration = 0
    # Define barrier subproblem
    subprob = BarrierSubproblem(
            x0, fun, grad, hess, constr, jac, constr_eq, jac_eq,
            lb, ub, A, b, A_eq, b_eq, barrier_parameter, tolerance,
            max_substep_iter)
    # Define initial parameter for the first iteration.
    z = subprob.z0()
    # Define trust region bounds
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf),
                          np.full(subprob.n_slack, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars+subprob.n_slack, np.inf)
    while True:
        # Update Barrier Problem
        subprob.update(barrier_parameter, tolerance)
        # Solve SQP subproblem
        z, info = equality_constrained_sqp(
            subprob.function,
            subprob.gradient,
            subprob.lagrangian_hessian,
            subprob.constraints,
            subprob.jacobian,
            z, v,
            trust_radius,
            trust_lb,
            trust_ub,
            subprob.stop_criteria,
            initial_penalty,
            subprob.scaling)

        # Update parameters
        iteration += info["niter"]
        trust_radius = max(initial_trust_radius,
                           TRUST_ENLARGEMENT*info["trust_radius"])
        v = info["v"]
        # TODO: Use more advanced strategies from [2]_
        # to update this parameters.
        barrier_parameter = BARRIER_DECAY_RATIO*barrier_parameter
        tolerance = BARRIER_DECAY_RATIO*tolerance
        # Update info
        info['niter'] = iteration

        if stop_criteria(info):
            # Get x
            x = subprob.get_variables(z)
            break

    return x, info

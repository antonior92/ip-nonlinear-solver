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
        subject to:  c_eq(x)       = 0
                     c_ineq(x) + s = 0
    """

    def __init__(self, fun, grad, lagr_hess,
                 c_eq, c_eq_jac, c_ineq, c_ineq_jac,
                 barrier_parameter, tolerance, max_iter,
                 n_vars, n_eq, n_ineq):
        self.fun = fun
        self.grad = grad
        self.lagr_hess_x = lagr_hess
        self.c_eq = c_eq
        self.c_eq_jac = c_eq_jac
        self.c_ineq = c_ineq
        self.c_ineq_jac = c_ineq_jac
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq

    def get_extended_param(self, x, s):
        return np.hstack((x, s))

    def get_slack(self, z):
        return z[self.n_vars:self.n_vars+self.n_ineq]

    def get_variables(self, z):
        return z[:self.n_vars]

    def get_eq_lagr_mult(self, v):
        return v[:self.n_eq]

    def get_ineq_lagr_mult(self, v):
        return v[self.n_eq:self.n_eq+self.n_ineq]

    def s0(self):
        return np.ones(self.n_ineq)

    def barrier_fun(self, z):
        """Returns barrier function at given funstion.

        For z = [x, s], returns barrier function:
            barrier_fun(z) = fun(x) - barrier_parameter*sum(log(s))
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        return self.fun(x) - self.barrier_parameter*np.sum(np.log(s))

    def constr(self, z):
        """Returns barrier problem constraints at given points.

        For z = [x, s], returns the constraints:

            constr(z) = [[ c_eq(x)       ]]
                        [[ c_ineq(x) + s ]].
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        c_eq = self.c_eq(x)
        c_ineq = self.c_ineq(x)
        return np.hstack((c_eq, c_ineq + s))

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
        return LinearOperator((self.n_vars+self.n_ineq,
                               self.n_vars+self.n_ineq),
                              matvec)

    def barrier_grad(self, z):
        """Returns scaled gradient (for the barrier problem).

        The result of scaling the gradient
        of the barrier problem by the previously
        defined scaling factor:
            barrier_grad = [[             grad(x)             ]]
                           [[ -barrier_parameter*ones(n_ineq) ]]
        """
        x = self.get_variables(z)
        return np.hstack((self.grad(x),
                          -self.barrier_parameter*np.ones(self.n_ineq)))

    def jac(self, z):
        """Returns scaled Jacobian.

        The result of scaling the Jacobian
        by the previously defined scaling factor:
            barrier_grad = [[ c_eq_jac(x)     0    ]]
                           [[ c_ineq_jac(x)  diag(s) ]]
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        S = spc.diags((s,), (0,))
        Aeq = self.c_eq_jac(x)
        Ain = self.c_ineq_jac(x)
        return spc.bmat([[Aeq, None], [Ain, S]], "csc")

    def lagr_hess_s(self, z, v):
        """Returns Lagrangian Hessian (in relation to slack variables ``s``)"""
        s = self.get_slack(z)
        v_ineq = self.get_ineq_lagr_mult(v)
        # Using the primal formulation:
        #     lagr_hess_s = diag(1/s**2).
        # Reference [1]_ p. 882, formula (3.1)
        primal = self.barrier_parameter/(s*s)
        # Using the primal-dual formulation
        #     lagr_hess_s = diag(v_ineq/s)
        # Reference [1]_ p. 883, formula (3.11)
        primal_dual = v_ineq/s
        # Uses the primal-dual formulation for
        # positives values of v_ineq, and primal
        # formulation for the remaining ones.
        return np.where(v_ineq > 0, primal_dual, primal)

    def lagr_hess(self, z, v):
        """Returns scaled Lagrangian Hessian"""
        x = self.get_variables(z)
        s = self.get_slack(z)
        # Compute Hessian in relation to x and s
        lagr_hess_x = self.lagr_hess_x
        Hx = lagr_hess_x(x, v)
        Hs = self.lagr_hess_s(z, v)*s*s

        # The scaled Lagragian Hessian is:
        #     [[ Hx    0    ]]
        #     [[ 0   S Hs S ]]
        def matvec(vec):
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            return np.hstack((Hx.dot(vec_x), Hs*vec_s))
        return LinearOperator((self.n_vars+self.n_ineq,
                               self.n_vars+self.n_ineq),
                              matvec)

    def stop_criteria(self, info):
        """Stop criteria to the barrier problem.

        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        if (info["opt"] < self.tolerance and info["constr_violation"] < self.tolerance) \
           or info["niter"] > self.max_iter:
            return True
        else:
            return False


def default_stop_criteria(info):
    if (info["opt"] < 1e-8 and info["constr_violation"] < 1e-8) \
       or info["niter"] > 1000:
        return True
    else:
        return False


def ipsolver(fun, grad, hess,
             c_ineq, c_ineq_jac,
             c_eq, c_eq_jac,
             x0, v0=None,
             stop_criteria=default_stop_criteria,
             initial_barrier_parameter=0.1,
             initial_tolerance=0.1,
             initial_penalty=1.0,
             initial_trust_radius=1.0,
             max_substep_iter=1000):
    """Trust-region interior points method.

    Solve problem:

        minimize fun(x)
        subject to: c_eq(x) = 0
                    c_ineq(x) <= 0

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

    n_vars, = np.shape(x0)  # Number of parameters
    n_eq, = np.shape(c_eq(x0))  # Number of equality constraints
    n_ineq, = np.shape(c_ineq(x0))  # Number of inequality constraitns
    # TODO: Avoid this unecessary call to the constraints.

    # Initial Values
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    v = v0
    z = None
    iteration = 0
    while True:
        # Define barrier subproblem
        subprob = BarrierSubproblem(
            fun, grad, hess, c_eq, c_eq_jac, c_ineq, c_ineq_jac,
            barrier_parameter, tolerance, max_substep_iter, n_vars,
            n_eq, n_ineq)
        # Define initial parameter for the first iteration.
        if z is None:
            z = subprob.get_extended_param(x0, subprob.s0())
        # Define trust region bounds
        trust_lb = np.hstack((np.full(n_vars, -np.inf),
                              np.full(n_ineq, -BOUNDARY_PARAMETER)))
        trust_ub = np.full(n_vars+n_ineq, np.inf)

        # Solve SQP subproblem
        z, info = equality_constrained_sqp(
            subprob.barrier_fun,
            subprob.barrier_grad,
            subprob.lagr_hess,
            subprob.constr,
            subprob.jac,
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
        v0 = info["v"]
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

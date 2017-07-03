"""Trust-region interior points methods"""

from __future__ import division, print_function, absolute_import
import scipy.sparse as spc
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator


class BarrierSubproblem:
    """
    Barrier optimization problem:

        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to:  c_eq(x)     = 0
                     c_ineq(x) + s = 0
    """

    def __init__(self, fun, grad, lagr_hess,
                 c_eq, c_eq_jac, c_ineq, c_ineq_jac,
                 barrier_parameter, tolerance,
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
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq

    def _get_slack(self, z):
        return z[self.n_vars:self.n+self.n_eq]

    def _get_variables(self, z):
        return z[:self.n_vars]

    def _get_eq_lagr_mult(self, v):
        return v[:n_eq]

    def _get_ineq_lagr_mult(self, v):
        return v[n_eq:n_eq+n_ineq]

    def barrier_fun(self, z):
        """Returns barrier function at given funstion.

        For z = [x, s], returns barrier function:
            barrier_fun(z) = fun(x) - barrier_parameter*sum(log(s))
        """
        x = self._get_variables(z)
        s = self._get_slack(z)
        return self.fun(x) - self.barrier_parameter*np.sum(np.log(s))

    def constr(self, z):
        """Returns barrier problem constraints at given points.

        For z = [x, s], returns the constraints:

            constr(z) = [[ c_eq(x)       ]]
                        [[ c_ineq(x) + s ]].
        """
        x = self._get_variables(z)
        s = self._get_slack(z)
        return np.hstack((self._c_eq(x), self._c_ineq(x) + s))

    def scaling(self, z):
        """Returns scaling vector.

        Given by:
            scaling = [ones(n_vars), s]
        """
        s = self._get_slack(z)
        return np.hstack((np.ones(n), s))

    def barrier_grad(self, z):
        """Returns scaled gradient (for the barrier problem).

        The result of scaling the gradient
        of the barrier problem by the previously
        defined scaling factor:
            barrier_grad = [[             grad(x)             ]]
                           [[ -barrier_parameter*ones(n_vars) ]]
        """
        x = self._get_variables(z)
        return np.hstack((self._grad(x), -barrier_parameter*np.ones(m_in)))

    def jac(self, z):
        """Returns scaled Jacobian.

        The result of scaling the Jacobian
        by the previously defined scaling factor:
            barrier_grad = [[ c_eq_jac(x)     0    ]]
                           [[ c_ineq_jac(x)  diag(s) ]]
        """
        x = self._get_variables(z)
        s = self._get_slack(z)
        S = spc.diags((s,), (0,))
        Aeq = self._c_eq_jac(x)
        Ain = self._c_ineq_jac(x)
        return spc.bmat([[Aeq, None], [Ain, S]], "csc")

    def lagr_hess_s(self, z, v):
        """Returns Lagrangian Hessian (in relation to slack variables ``s``)"""
        s = self._get_slack(z)
        v_ineq = self._get_ineq_lagr_mult(v)
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
        x = self._get_variables(z)
        s = self._get_slack(z)
        # Compute Hessian in relation to x and s
        Hx = self.lagr_hess_x(z, v)
        Hs = self.lagr_hess_s(z, v)*s*s

        # The scaled Lagragian Hessian is:
        #     [[ Hx    0    ]]
        #     [[ 0   S Hs S ]]
        def matvec(vec):
            vec_x = self._get_variables(vec)
            vec_s = self._get_slack(vec)
            return np.hstack((Hx.dot(vec_x), Hs*vec_s))
        return LinearOperator((n_vars+n_ineq, n_vars+n_ineq), matvec)

    def stop_criteria(self, z, v, fun, grad, constr, jac, iteration,
                      trust_radius):
        """Stop criteria to the barrier problem.

        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        opt = norm(grad + jac.T.dot(v), np.inf)
        c_violation = norm(constr, np.inf)
        if opt < self.tolerance \
           and c_violation < self.tolerance:
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
             initial_trust_radius=1.0):
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
    v0 = None
    iteration = 0

    while not stop_criteria():
        # Define barrier subproblem
        subprob = BarrierSubproblem(
            fun, grad, hess, c_eq, c_eq_jac, c_ineq, c_ineq_jac,
            barrier_parameter, tolerance, n_vars, n_eq, n_ineq)
        # Define trust region bounds
        trust_lb = np.hstack(np.full(n_vars, -np.inf),
                             np.full(n_ineq, -BOUNDARY_PARAMETER))
        trust_ub = np.full(n_vars+n_ineq, np.inf)

        # Solve SQP subproblem
        x, info = equality_constrained_sqp(
            subprob.barrier_fun,
            subprob.barrier_grad,
            subprob.lagr_hess,
            subprob.constr,
            subprob.jac,
            x0, v0,
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

    info['niter'] = iteration
    return x, info

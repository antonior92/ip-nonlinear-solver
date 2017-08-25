# ip-nonlinear-solver

[![Join the chat at https://gitter.im/ip-nonlinear-solver/GSoC2017](https://badges.gitter.im/ip-nonlinear-solver/GSoC2017.svg)](https://gitter.im/ip-nonlinear-solver/GSoC2017?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A trust-region interior-point method for general nonlinear programing problems. The implemetation
is part my GSoC project for Scipy. A series of blog post describe different aspects of the algorithm
and its use  ([link](https://antonior92.github.io/tags/#gsoc-2017)).

The implementation can be found on the separate repository and is being
integrated to SciPy through the pull request:

[https://github.com/scipy/scipy/pull/7729](https://github.com/scipy/scipy/pull/7729)


## Instalation Guide

1) **Download repository from github**

```bash
git clone git@github.com:antonior92/ip-nonlinear-solver.git
# or, alternatively:
# git clone https://github.com/antonior92/ip-nonlinear-solver.git
```

2) **Install package for development**

Install package for development with:
```bash
python setup.py develop
```

or, if you need root user priveleges:
```bash
sudo python setup.py develop
```

## Usage Example
Consider the following minimization problem:
```
min (1/2)*(x -2)**2 + (1/2)*(y - 1/2)**2, 
subject to: 1/(x + 1) - y >= 1/4;
            x >= 0
            y >= 0 
```

The code for solving this problem is presented bellow:

```Python
# Example
from __future__ import division
import numpy as np
from ipsolver import minimize_constrained, NonlinearConstraint, BoxConstraint

# Define objective function and derivatives
fun = lambda x: 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2
grad = lambda x: np.array([x[0] - 2, x[1] - 1/2])
hess =  lambda x: np.eye(2)
# Define nonlinear constraint
c = lambda x: np.array([1/(x[0] + 1) - x[1],])
c_jac = lambda x: np.array([[-1/(x[0] + 1)**2, -1]])
c_hess = lambda x, v: 2*v[0]*np.array([[1/(x[0] + 1)**3, 0], [0, 0]])
nonlinear = NonlinearConstraint(c, ('greater', 1/4), c_jac, c_hess)
# Define box constraint
box = BoxConstraint(("greater",))

# Define initial point
x0 = np.array([0, 0])
# Apply solver
result = minimize_constrained(fun, x0, grad, hess, (nonlinear, box))
```

## Documentation

The function [``minimize_constrained``](https://github.com/antonior92/ip-nonlinear-solver#minimize_constrained) solves nonlinear programming problems.
This functions minimizes an object function subject to constraints. These constraints
can be specified using the classes [``NonlinearConstraint``](https://github.com/antonior92/ip-nonlinear-solver#NonlinearConstraint),
[``LinearConstraint``](https://github.com/antonior92/ip-nonlinear-solver#LinearConstraint) and [``BoxConstraint``](https://github.com/antonior92/ip-nonlinear-solver#BoxConstraint).


### ``minimize_constrained``

```
minimize_constrained(fun, x0, grad, hess='2-point', constraints=(),
                     method=None, xtol=1e-8, gtol=1e-8, sparse_jacobian=None,
                     options={}, callback=None, max_iter=1000, verbose=0)

    Minimize scalar function subject to constraints.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            fun(x) -> float

        where x is an array with shape (n,).
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    grad : callable
        Gradient of the objective function:

            grad(x) -> array_like, shape (n,)

        where x is an array with shape (n,).
    hess : {callable, '2-point', '3-point', 'cs', None}, optional
        Method for computing the Hessian matrix. The keywords
        select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as much operations compared to '2-point' (default). The
        scheme 'cs' uses complex steps, and while potentially the most
        accurate, it is applicable only when `fun` correctly handles
        complex inputs and can be analytically continued to the complex
        plane. If it is a callable, it should return the 
        Hessian matrix of `dot(fun, v)`:

            hess(x, v) -> {LinearOperator, sparse matrix, ndarray}, shape (n, n)

        where x is a (n,) ndarray and v is a (m,) ndarray. When ``hess``
        is None it considers the hessian is an matrix filled with zeros.
    constraints : Constraint or List of Constraint's, optional
        A single object or a list of objects specifying
        constraints to the optimization problem.
        Available constraints are:

            - `BoxConstraint`
            - `LinearConstraint`
            - `NonlinearConstraint`

    method : {str, None}, optional
        Type of solver. Should be one of:

            - 'equality-constrained-sqp'
            - 'tr-interior-point'

        When None the more appropriate method is choosen.
        'equality-constrained-sqp' is chosen for problems that
        only have equality constraints and 'tr-interior-point'
        for general optimization problems.
    xtol : float, optional
        Tolerance for termination by the change of the independent variable.
        The algorithm will terminate when ``delta < xtol``, where ``delta``
        is the algorithm trust-radius. Default is 1e-8.
    gtol : float, optional
        Tolerance for termination by the norm of the Lagrangian gradient.
        The algorithm will terminate when both the infinity norm (i.e. max
        abs value) of the Lagrangian gradient and the constraint violation
        are smaller than ``gtol``. Default is 1e-8.
    sparse_jacobian : {bool, None}
        The algorithm uses a sparse representation of the Jacobian if True
        and a dense representation if False. When sparse_jacobian is None
        the algorithm uses the more convenient option, using a sparse
        representation if at least one of the constraint Jacobians are sparse
        and a dense representation when they are all dense arrays.
    options : dict, optional
        A dictionary of solver options. Available options include:

            initial_trust_radius: float
                Initial trust-region radius. By defaut uses 1.0, as
                suggested in [1]_, p.19, immediatly after algorithm III.
            initial_penalty : float
                Initial penalty for merit function. By defaut uses 1.0, as
                suggested in [1]_, p.19, immediatly after algorithm III.
            initial_barrier_parameter: float
                Initial barrier parameter. Exclusive for 'tr_interior_point'
                method. By default uses 0.1, as suggested in [1]_ immediatly
                after algorithm III, p. 19.
            initial_tolerance: float
                Initial subproblem tolerance. Exclusive for
                'tr_interior_point' method. By defaut uses 0.1,
                as suggested in [1]_ immediatly after algorithm III, p. 19.
            return_all : bool, optional
                When True return the list of all vectors
                through the iterations.
            factorization_method : string, optional
                Method used for factorizing the jacobian matrix.
                Should be one of:

                - 'NormalEquation': The operators
                   will be computed using the
                   so-called normal equation approach
                   explained in [1]_. In order to do
                   so the Cholesky factorization of
                   ``(A A.T)`` is computed. Exclusive
                   for sparse matrices. Requires
                   scikit-sparse installed.
                - 'AugmentedSystem': The operators
                   will be computed using the
                   so-called augmented system approach
                   explained in [1]_. It perform the
                   LU factorization of an augmented
                   system. Exclusive for sparse matrices.
                - 'QRFactorization': Compute projections
                   using QR factorization. Exclusive for
                   dense matrices.
                - 'SVDFactorization': Compute projections
                   using SVD factorization. Exclusive for
                   dense matrices.

                The factorization methods 'NormalEquation' and
                'AugmentedSystem' should be used only when
                ``sparse_jacobian=True``. They usually provide
                similar results. The methods 'QRFactorization'
                and 'SVDFactorization' should be used when
                ``sparse_jacobian=False``. By default uses
                'QRFactorization' for  dense matrices.
                The 'SVDFactorization' method can cope
                with Jacobian matrices with deficient row
                rank and will be used whenever other
                factorization methods fails (which may
                imply the conversion to a dense format).

    callback : callable, optional
        Called after each iteration:

            callback(OptimizeResult state) -> bool

        If callback returns True the algorithm execution is terminated.
        ``state`` is an `OptimizeResult` object, with the same fields
        as the ones from the return.
    max_iter : int, optional
        Maximum number of algorithm iterations. By default ``max_iter=1000``
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    `OptimizeResult` with the following fields defined:
    x : ndarray, shape (n,)
        Solution found.
    s : ndarray, shape (n_ineq,)
        Slack variables at the solution. ``n_ineq`` is the total number
        of inequality constraints.
    v : ndarray, shape (n_ineq + n_eq,)
        Estimated Lagrange multipliers at the solution. ``n_ineq + n_eq``
        is the total number of equality and inequality constraints.
    niter : int
        Total number of iterations.
    nfev : int
        Total number of objective function evaluations.
    ngev : int
        Total number of objective function gradient evaluations.
    nhev : int
        Total number of Lagragian Hessian evaluations. Each time the
        Lagrangian Hessian is evaluated the objective function
        Hessian and the constraints Hessians are evaluated
        one time each.
    ncev : int
        Total number of constraint evaluations. The same couter
        is used for equality and inequality constraints, because
        they always are evaluated the same number of times.
    njev : int
        Total number of constraint Jacobian matrix evaluations.
        The same couter is used for equality and inequality
        constraint Jacobian matrices, because they always are
        evaluated the same number of times.
    cg_niter : int
        Total number of CG iterations.
    cg_info : Dict
        Dictionary containing information about the latest CG iteration:

            - 'niter' : Number of iterations.
            - 'stop_cond' : Reason for CG subproblem termination:

                1. Iteration limit was reached;
                2. Reached the trust-region boundary;
                3. Negative curvature detected;
                4. Tolerance was satisfied.

            - 'hits_boundary' : True if the proposed step is on the boundary
              of the trust region.

    execution_time : float
        Total execution time.
    trust_radius : float
        Trust radius at the last iteration.
    penalty : float
        Penalty function at last iteration.
    tolerance : float
        Tolerance for barrier subproblem at the last iteration.
        Exclusive for 'tr_interior_point'.
    barrier_parameter : float
        Barrier parameter at the last iteration. Exclusive for
        'tr_interior_point'.
    status : {0, 1, 2, 3}
        Termination status:

            * 0 : The maximum number of function evaluations is exceeded.
            * 1 : `gtol` termination condition is satisfied.
            * 2 : `xtol` termination condition is satisfied.
            * 3 : `callback` function requested termination.

    message : str
        Termination message.
    method : {'equality_constrained_sqp', 'tr_interior_point'}
        Optimization method used.
    constr_violation : float
        Constraint violation at last iteration.
    optimality : float
        Norm of the Lagrangian gradient at last iteration.
    fun : float
        For the 'equality_constrained_sqp' method this is the objective
        function evaluated at the solution and for the 'tr_interior_point'
        method this is the barrier function evaluated at the solution.
    grad : ndarray, shape (n,)
        For the 'equality_constrained_sqp' method this is the gradient of the
        objective function evaluated at the solution and for the
        'tr_interior_point' method  this is the gradient of the barrier
        function evaluated at the solution.
    constr : ndarray, shape (n_ineq + n_eq,)
        For the 'equality_constrained_sqp' method this is the equality
        constraint evaluated at the solution and for the 'tr_interior_point'
        method this are the equality and inequality constraints evaluated at
        a given point (with the inequality constraints incremented by the
        value of the slack variables).
    jac : {ndarray, sparse matrix}, shape (n_ineq + n_eq, n)
        For the 'equality_constrained_sqp' method this is the Jacobian
        matrix of the equality constraint evaluated at the solution and
        for the tr_interior_point' method his is scaled augmented Jacobian
        matrix, defined as ``\hat(A)`` in equation (19.36), reference [2]_,
        p. 581.

    Notes
    -----
    Method 'equality_constrained_sqp' is an implementation of
    Byrd-Omojokun Trust-Region SQP method described [3]_ and
    in [2]_, p. 549. It solves equality constrained equality
    constrained optimization problems by solving, at each substep,
    a trust-region QP subproblem. The inexact solution of these
    QP problems using projected CG method makes this method
    appropriate for large-scale problems.

    Method 'tr_interior_point' is an implementation of the
    trust-region interior point method described in [1]_.
    It solves general nonlinear by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is used
    to solve the subproblems with increasing levels of accuracy as
    the iterate gets closer to a solution. It is also an
    appropriate method for large-scale problems.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [2] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    .. [3] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. "On the
           implementation of an algorithm for large-scale equality
           constrained optimization." SIAM Journal on
           Optimization 8.3 (1998): 682-706.
```

### ``NonlinearConstraint``

```
NonlinearConstraint(fun, kind, jac, hess='2-point', enforce_feasibility=False)

   Nonlinear constraint

    Parameters
    ----------
    fun : callable
        The function defining the constraint.

            fun(x) -> array_like, shape (m,)

        where x is a (n,) ndarray and ``m``
        is the number of constraints.
    kind : {str, tuple}
        Specifies the type of contraint. Options for this
        parameter are:

            - ('interval', lb, ub) for a constraint of the type:
                lb <= fun(x) <= ub
            - ('greater', lb) for a constraint of the type:
                fun(x) >= lb
            - ('less', ub) for a constraint of the type:
                fun(x) <= ub
            - ('equals', c) for a constraint of the type:
                fun(x) == c
            - ('greater',) for a constraint of the type:
                fun(x) >= 0
            - ('less',) for a constraint of the type:
                fun(x) <= 0
            - ('equals',) for a constraint of the type:
                fun(x) == 0

        where ``lb``,  ``ub`` and ``c`` are (m,) ndarrays or
        scalar values. In the latter case, the same value
        will be repeated for all the constraints.
    jac : callable
        Jacobian Matrix:

            jac(x) -> {ndarray, sparse matrix}, shape (m, n)

        where x is a (n,) ndarray.
    hess : {callable, '2-point', '3-point', 'cs', None}
        Method for computing the Hessian matrix. The keywords
        select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as much operations compared to '2-point' (default). The
        scheme 'cs' uses complex steps, and while potentially the most
        accurate, it is applicable only when `fun` correctly handles
        complex inputs and can be analytically continued to the complex
        plane. If it is a callable, it should return the 
        Hessian matrix of `dot(fun, v)`:

            hess(x, v) -> {LinearOperator, sparse matrix, ndarray}, shape (n, n)

        where x is a (n,) ndarray and v is a (m,) ndarray. When ``hess``
        is None it considers the hessian is an matrix filled with zeros.
    enforce_feasibility : {list of boolean, boolean}, optional
        Specify if the constraint must be feasible along the iterations.
        If ``True``  all the iterates generated by the optimization
        algorithm need to be feasible in respect to a constraint. If ``False``
        this is not needed. A list can be passed to specify element-wise
        each constraints needs to stay feasible along the iterations and
        each does not. Alternatively, a single boolean can be used to
        specify the feasibility required of all constraints. By default it
        is False.
```

### ``LinearConstraint``

```
LinearConstraint(A, kind, enforce_feasibility=False)

    Linear constraint.

    Parameters
    ----------
    A : {ndarray, sparse matrix}, shape (m, n)
        Matrix for the linear constraint.
    kind : {str, tuple}
        Specifies the type of contraint. Options for this
        parameter are:

            - ('interval', lb, ub) for a constraint of the type:
                lb <= A x <= ub
            - ('greater', lb) for a constraint of the type:
                A x >= lb
            - ('less', ub) for a constraint of the type:
                A x <= ub
            - ('equals', c) for a constraint of the type:
                A x == c
            - ('greater',) for a constraint of the type:
                A x >= 0
            - ('less',) for a constraint of the type:
                A x <= 0
            - ('equals',) for a constraint of the type:
                A x == 0

        where ``lb``,  ``ub`` and ``c`` are (m,) ndarrays or
        scalar values. In the latter case, the same value
        will be repeated for all the constraints.
    enforce_feasibility : {list of boolean, boolean}, optional
        Specify if the constraint must be feasible along the iterations.
        If ``True``  all the iterates generated by the optimization
        algorithm need to be feasible in respect to a constraint. If ``False``
        this is not needed. A list can be passed to specify element-wise
        each constraints needs to stay feasible along the iterations and
        each does not. Alternatively, a single boolean can be used to
        specify the feasibility required of all constraints. By default it
        is False.
```

### <a name="BoxConstraint"></a> ``BoxConstraint``

```
BoxConstraint(kind, enforce_feasibility=False)

    Box constraint.

    Parameters
    ----------
    kind : tuple
        Specifies the type of contraint. Options for this
        parameter are:

            - ('interval', lb, ub) for a constraint of the type:
                lb <= A x <= ub
            - ('greater', lb) for a constraint of the type:
                A x >= lb
            - ('less', ub) for a constraint of the type:
                A x <= ub
            - ('equals', c) for a constraint of the type:
                A x == c
            - ('greater',) for a constraint of the type:
                A x >= 0
            - ('less',) for a constraint of the type:
                A x <= 0
            - ('equals',) for a constraint of the type:
                A x == 0

        where ``lb``,  ``ub`` and ``c`` are (m,) ndarrays or
        scalar values. In the latter case, the same value
        will be repeated for all the constraints.
    enforce_feasibility : {list of boolean, boolean}, optional
        Specify if the constraint must be feasible along the iterations.
        If ``True``  all the iterates generated by the optimization
        algorithm need to be feasible in respect to a constraint. If ``False``
        this is not needed. A list can be passed to specify element-wise
        each constraints needs to stay feasible along the iterations and
        each does not. Alternatively, a single boolean can be used to
        specify the feasibility required of all constraints. By default it
        is False.
```

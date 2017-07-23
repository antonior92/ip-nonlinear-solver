import numpy as np
from ipsolver import (ipsolver,
                      opt_problems,
                      parse_matlab_like_problem)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestIPSolver(TestCase):

    def test_problems(self):

        list_of_problems = [opt_problems.Rosenbrock(),
                            opt_problems.Rosenbrock(n=20),
                            opt_problems.BoundContrRosenbrock(),
                            opt_problems.BoundContrRosenbrock(n=20),
                            opt_problems.IneqLinearConstrRosenbrock(),
                            opt_problems.LinearConstrRosenbrock(),
                            opt_problems.SimpleIneqConstr()]

        for p in list_of_problems:
            fun, grad, lagr_hess, n_ineq, constr_ineq, \
                jac_ineq, n_eq, constr_eq, jac_eq \
                = parse_matlab_like_problem(
                    p.fun, p.grad, p.lagr_hess, p.n_vars, p.n_eq, p.n_ineq,
                    p.constr_ineq, p.jac_ineq, p.constr_eq, p.jac_eq, p.A_ineq,
                    p.b_ineq, p.A_eq, p.b_eq, p.lb, p.ub)
            x, info = ipsolver(
                fun, grad, lagr_hess, n_ineq, constr_ineq,
                jac_ineq, n_eq, constr_eq, jac_eq, p.x0)
            assert_array_less(info["opt"], 1e-5)
            assert_array_less(info["constr_violation"], 1e-8)
            if p.x_opt is not None:
                assert_array_almost_equal(x, p.x_opt, decimal=5)

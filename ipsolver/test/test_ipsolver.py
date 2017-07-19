import numpy as np
from ipsolver import (ipsolver,
                      ProblemSimpleIneqConstr,
                      ProblemELEC,
                      ProblemMaratos,
                      ProblemRosenbrock,
                      ProblemBoundContrRosenbrock,
                      ProblemIneqLinearConstrRosenbrock,
                      ProblemLinearConstrRosenbrock)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestIPSolver(TestCase):

    def test_problems(self):

        list_of_problems = [ProblemRosenbrock(),
                            ProblemRosenbrock(n=20),
                            ProblemBoundContrRosenbrock(),
                            ProblemBoundContrRosenbrock(n=20),
                            ProblemIneqLinearConstrRosenbrock(),
                            ProblemLinearConstrRosenbrock(),
                            ProblemSimpleIneqConstr(),
                            ProblemMaratos(),
                            ProblemELEC(n_electrons=50)]

        for p in list_of_problems:
            x, info = ipsolver(
                p.fun, p.grad, p.lagr_hess, p.x0, p.constr_ineq, p.jac_ineq,
                p.constr_eq, p.jac_eq, p.A_ineq, p.b_ineq, p.A_eq, p.b_ineq,
                p.lb, p.ub)
            assert_array_less(info["opt"], 1e-5)
            assert_array_less(info["constr_violation"], 1e-8)
            if p.x_opt is not None:
                assert_array_almost_equal(x, p.x_opt, decimal=5)

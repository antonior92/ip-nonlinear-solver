import numpy as np
from ipsolver import (equality_constrained_sqp,
                      ProblemMaratos,
                      ProblemELEC)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestEqualityConstrainedSQPSolver(TestCase):

    def test_on_equality_constrained_problems(self):

        list_of_problems = [ProblemMaratos(degrees=60),
                            ProblemMaratos(degrees=10),
                            ProblemELEC(n_electrons=10),
                            ProblemELEC(n_electrons=30),
                            ProblemELEC(n_electrons=50)]

        for p in list_of_problems:
            x, info = equality_constrained_sqp(
                p.fun, p.grad, p.lagr_hess, p.constr_eq, p.jac_eq, p.x0,
                p.v0, initial_trust_radius=1, initial_penalty=1,
                return_all=True)
            assert_array_less(info["opt"], 1e-5)
            assert_array_less(info["constr_violation"], 1e-8)
            if p.x_opt is not None:
                assert_array_almost_equal(x, p.x_opt)
            if p.f_opt is not None:
                assert_array_almost_equal(x, p.x_opt)

import numpy as np
from ipsolver import (ipsolver,
                      ProblemSimpleIneqConstr)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestIPSolver(TestCase):

    def test_on_constrained_problems(self):

        list_of_problems = [ProblemSimpleIneqConstr(), ]

        for p in list_of_problems:
            x, info = ipsolver(
                p.fun, p.grad, p.lagr_hess, p.x0, p.constr, p.jac)
            assert_array_less(info["opt"], 1e-8)
            assert_array_less(info["constr_violation"], 1e-8)
            if p.x_opt is not None:
                assert_array_almost_equal(x, p.x_opt)

import numpy as np
from scipy.sparse import csc_matrix
from ipsolver import eqp_kktfact
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestEQPDirectFactorization(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452
    def test_eqp_kktfact(self):

        G = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = np.array([3, 0])

        x, lagrange_multipliers = eqp_kktfact(G, c, A, b)

        assert_array_almost_equal(x, [2, -1, 1])
        assert_array_almost_equal(lagrange_multipliers, [3, -2])

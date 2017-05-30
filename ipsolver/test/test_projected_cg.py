import numpy as np
from scipy.sparse import csc_matrix
from ipsolver import eqp_kktfact, projections, projected_cg
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestEQPDirectFactorization(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452
    def test_nocedal_example(self):

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


class TestProjections(TestCase):

    def test_nullspace_and_least_squares(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)

        Z, LS, _ = projections(A)

        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for z in test_points:
            # Test if x is in the null_space
            x = Z.matvec(z)
            assert_array_almost_equal(A.dot(x), 0)

            # Test if x is the least square solution
            x = LS.matvec(z)
            x2 = np.linalg.lstsq(At_dense, z)[0]
            assert_array_almost_equal(x, x2)

    def test_rowspace(self):

        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)

        _, _, Y = projections(A)

        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        for z in test_points:
            # Test if x is solution of A x = z
            x = Y.matvec(z)
            assert_array_almost_equal(A.dot(x), z)

            # Test if x is in the row space of A
            A_ext = np.vstack((A_dense, x))
            assert_equal(np.linalg.matrix_rank(A_dense),
                         np.linalg.matrix_rank(A_ext))


class TestProjectCG(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452
    def test_nocedal_example(self):

        G = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = np.array([3, 0])

        Z, _, Y = projections(A)

        x = projected_cg(G, c, Z, Y, b)

        assert_array_almost_equal(x, [2, -1, 1])

    def test_compare_with_direct_fact(self):

        G = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 10]])
        c = np.array([-8, -3, -3, 10])
        b = np.array([3, 0])

        Z, _, Y = projections(A)

        x = projected_cg(G, c, Z, Y, b, tol=1e-15)
        x_kkt, _ = eqp_kktfact(G, c, A, b)

        assert_array_almost_equal(x, x_kkt)

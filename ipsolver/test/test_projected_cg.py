import numpy as np
from scipy.sparse import csc_matrix
from ipsolver import eqp_kktfact, projections, projected_cg, orthogonality
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestEQPDirectFactorization(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452
    def test_nocedal_example(self):

        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = np.array([3, 0])

        x, lagrange_multipliers = eqp_kktfact(H, c, A, b)

        assert_array_almost_equal(x, [2, -1, 1])
        assert_array_almost_equal(lagrange_multipliers, [3, -2])


class TestProjections(TestCase):

    def test_nullspace_and_least_squares(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)

        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for method in ("NormalEquation", "AugmentedSystem"):
            Z, LS, _ = projections(A, method)

            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_array_almost_equal(A.dot(x), 0)

                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0)

                # Test if x is the least square solution
                x = LS.matvec(z)
                x2 = np.linalg.lstsq(At_dense, z)[0]
                assert_array_almost_equal(x, x2)

    def test_iterative_refinements(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)

        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        for method in ("NormalEquation", "AugmentedSystem"):
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=100)

            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_array_almost_equal(A.dot(x), 0, decimal=14)

                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0, decimal=16)

    def test_rowspace(self):

        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)

        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        for method in ('NormalEquation', 'AugmentedSystem'):
            _, _, Y = projections(A, method)

            for z in test_points:
                # Test if x is solution of A x = z
                x = Y.matvec(z)
                assert_array_almost_equal(A.dot(x), z)

                # Test if x is in the return row space of A
                A_ext = np.vstack((A_dense, x))
                assert_equal(np.linalg.matrix_rank(A_dense),
                             np.linalg.matrix_rank(A_ext))


class TestOrthogonality(TestCase):

    def test_dense_matrix(self):

        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])

        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806,   2935.29289083])

        test_expected_orth = (0, 0)

        for i in range(len(test_vectors)):
            x = test_vectors[i]
            orth = test_expected_orth[i]

            assert_array_almost_equal(orthogonality(A, x), orth)

    def test_sparse_matrix(self):

        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A)

        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806,   2935.29289083])

        test_expected_orth = (0, 0)

        for i in range(len(test_vectors)):
            x = test_vectors[i]
            orth = test_expected_orth[i]

            assert_array_almost_equal(orthogonality(A, x), orth)


class TestProjectCG(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452
    def test_nocedal_example(self):

        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = np.array([3, 0])

        Z, _, Y = projections(A)

        x, info = projected_cg(H, c, Z, Y, b)

        assert_array_almost_equal(x, [2, -1, 1])

    def test_compare_with_direct_fact(self):

        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = np.array([3, 0])

        Z, _, Y = projections(A)

        x, info = projected_cg(H, c, Z, Y, b, tol=0)
        x_kkt, _ = eqp_kktfact(H, c, A, b)

        assert_array_almost_equal(x, x_kkt)

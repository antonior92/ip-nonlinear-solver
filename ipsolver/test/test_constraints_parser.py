import numpy as np
import scipy.sparse as spc
from ipsolver import (parse_linear_constraints,
                      parse_nonlinear_constraints)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestParseLinearConstraints(TestCase):

    def test_for_dense_matrix(self):
        lb = np.array([1, -np.inf, 2, 3])
        ub = np.array([np.inf, 1, 2, np.inf])
        A = np.array([[1, 1, 1, 1],
                       [2, 2, 2, 2],
                       [3, 3, 3, 3],
                       [4, 4, 4, 4]])

        A_ineq0 = np.array([[2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [-1, -1, -1, -1],
                            [-3, -3, -3, -3],
                            [-4, -4, -4, -4]])
        b_ineq0 = np.array([1, 2, -1, -2, -3])

        A_ineq, b_ineq = parse_linear_constraints(A, lb, ub)
        assert_array_equal(A_ineq, A_ineq0)
        assert_array_equal(b_ineq, b_ineq0)

    def test_for_sparse_matrix(self):
        lb = np.array([1, -np.inf, 2, 3])
        ub = np.array([np.inf, 1, 2, np.inf])
        A = spc.csc_matrix([[1, 0, 0, 0],
                            [0, 2, 0, 2],
                            [0, 0, 0, 3],
                            [4, 0, 4, 4]])

        A_ineq0 = np.array([[0, 2, 0, 2],
                            [0, 0, 0, 3],
                            [-1, 0, 0, 0],
                            [0, 0, 0, -3],
                            [-4, 0, -4, -4]])
        b_ineq0 = np.array([1, 2, -1, -2, -3])

        A_ineq, b_ineq = parse_linear_constraints(A, lb, ub)
        assert_array_equal(A_ineq.toarray(), A_ineq0)
        assert_array_equal(b_ineq, b_ineq0)


class TestParseNonlinearConstraints(TestCase):

    def test_for_linear_constraint(self):
        lb = np.array([1, -np.inf, 2, 3])
        ub = np.array([np.inf, 1, 2, np.inf])
        A1 = np.array([[1, 0, 0, 0],
                       [0, 2, 0, 0],
                       [0, 0, 3, 0],
                       [0, 0, 0, 4]])
        A2 = spc.csc_matrix(A1)

        for A in [A1, A2]:
            def constraint(x):
                return A.dot(x)

            def jacobian(x):
                return A

            constr_ineq, jac_ineq \
                = parse_nonlinear_constraints(constraint, jacobian, lb, ub)

            A_ineq0 = np.array([[0, 2, 0, 0],
                                [0, 0, 3, 0],
                                [-1, 0, 0, 0],
                                [0, 0, -3, 0],
                                [0, 0, 0, -4]])
            b_ineq0 = np.array([1, 2, -1, -2, -3])

            def constr_ineq0(x):
                return np.dot(A_ineq0, x) - b_ineq0

            def jac_ineq0(x):
                return A_ineq0

            test_points = [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]
            for x in test_points:
                assert_array_equal(constr_ineq(x), constr_ineq0(x))
                aux = jac_ineq(x)
                if spc.issparse(aux):
                    assert_array_equal(aux.toarray(), jac_ineq0(x))
                else:
                    assert_array_equal(aux, jac_ineq0(x))

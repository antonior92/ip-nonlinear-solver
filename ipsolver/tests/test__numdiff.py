from __future__ import division

import math
from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from ipsolver._numdiff import approx_derivative


class TestApproxDerivativeLinearOperator(object):

    def fun_scalar_scalar(self, x):
        return np.sinh(x)

    def jac_scalar_scalar(self, x):
        return np.cosh(x)

    def fun_scalar_vector(self, x):
        return np.array([x[0]**2, np.tan(x[0]), np.exp(x[0])])

    def jac_scalar_vector(self, x):
        return np.array(
            [2 * x[0], np.cos(x[0]) ** -2, np.exp(x[0])]).reshape(-1, 1)

    def fun_vector_scalar(self, x):
        return np.sin(x[0] * x[1]) * np.log(x[0])

    def jac_vector_scalar(self, x):
        return np.array([
            x[1] * np.cos(x[0] * x[1]) * np.log(x[0]) +
            np.sin(x[0] * x[1]) / x[0],
            x[0] * np.cos(x[0] * x[1]) * np.log(x[0])
        ])

    def fun_vector_vector(self, x):
        return np.array([
            x[0] * np.sin(x[1]),
            x[1] * np.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def jac_vector_vector(self, x):
        return np.array([
            [np.sin(x[1]), x[0] * np.cos(x[1])],
            [-x[1] * np.sin(x[0]), np.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def test_scalar_scalar(self):
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_scalar_scalar(x0)
        np.random.seed(1)
        for i in range(10):
            p = np.random.uniform(-10, 10, size=(1,))
            assert_allclose(jac_diff_2.dot(p), jac_true*p,
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true*p,
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true*p,
                            rtol=5e-6)

    def test_scalar_vector(self):
        x0 = 0.5
        jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_scalar_vector(np.atleast_1d(x0))
        np.random.seed(1)
        for i in range(10):
            p = np.random.uniform(-10, 10, size=(1,))
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p),
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p),
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p),
                            rtol=5e-6)

    def test_vector_scalar(self):
        x0 = np.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_vector_scalar(x0)
        np.random.seed(1)
        for i in range(10):
            p = np.random.uniform(-10, 10, size=x0.shape)
            assert_allclose(jac_diff_2.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), np.atleast_1d(jac_true.dot(p)),
                            rtol=1e-7)

    def test_vector_vector(self):
        x0 = np.array([-100.0, 0.2])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_vector_vector(x0)
        np.random.seed(1)
        for i in range(10):
            p = np.random.uniform(-10, 10, size=x0.shape)
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p), rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p), rtol=1e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p), rtol=1e-7)

    def test_exception(self):
        x0 = np.array([-100.0, 0.2])
        assert_raises(ValueError, approx_derivative,
                      self.fun_vector_vector, x0,
                      method='2-point', bounds=(1, np.inf))

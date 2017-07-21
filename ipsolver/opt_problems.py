"""Constrained optimization problems in Native Python"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix


__all__ = ['Maratos',
           'SimpleIneqConstr',
           'ELEC',
           'Rosenbrock',
           'BoundContrRosenbrock',
           'IneqLinearConstrRosenbrock',
           'LinearConstrRosenbrock']


class MatlabLikeInterface(object):
    """Matlab like optimization problem interface.

    Optimization problem of the form:

        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
                    A_ineq x <= b_ineq
                    A_eq x = b_eq
                    lb <= x <= ub

    """
    constr_eq = None
    jac_eq = None
    hess_eq = None
    constr_ineq = None
    jac_ineq = None
    hess_ineq = None
    lb = None
    ub = None
    A_eq = None
    b_eq = None
    A_ineq = None
    b_ineq = None
    x_opt = None
    f_opt = None

    def lagr_hess(self, x, v_eq, v_ineq=None):
        H = self.hess(x)
        if self.hess_eq is not None:
            H += self.hess_eq(x, v_eq)
        if self.hess_ineq is not None:
            H += self.hess_ineq(x, v_ineq)
        return H


class Maratos(MatlabLikeInterface):
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60):
        rads = degrees/180*np.pi
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.v0 = [0]
        self.x_opt = np.array([1.0, 0.0])
        self.f_opt = -1.0

    def fun(self, x):
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x):
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x):
        return 4*np.eye(2)

    def constr_eq(self, x):
        return np.array([x[0]**2 + x[1]**2 - 1])

    def jac_eq(self, x):
        return np.array([[4*x[0], 4*x[1]]])

    def hess_eq(self, x, v):
        return 2*v[0]*np.eye(2)


class SimpleIneqConstr(MatlabLikeInterface):
    """Problem 15.1 from Nocedal and Wright

    The following optimization problem:
        minimize 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2
        Subject to: -1/(x[0] + 1) + x[1] + 1/4 <= 0
                                        - x[0] <= 0
                                        - x[1] <= 0
    """

    def __init__(self):
        self.x0 = [0, 0]
        self.v0 = [0]
        self.x_opt = np.array([1.952823,  0.088659])
        self.f_opt = 0.08571

    def fun(self, x):
        return 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2

    def grad(self, x):
        return np.array([x[0] - 2, x[1] - 1/2])

    def hess(self, x):
        return np.eye(2)

    def constr_ineq(self, x):
        return np.array([-1/(x[0] + 1) + x[1] + 1/4, -x[0], -x[1]])

    def jac_ineq(self, x):
        return np.array([[1/(x[0] + 1)**2, 1],
                         [-1, 0],
                         [0, -1]])

    def hess_ineq(self, x, v):
        return 2*v[0]*np.array([[1/(x[0] + 1)**3, 0],
                                [0, 0]])


class ELEC(MatlabLikeInterface):
    def __init__(self, n_electrons=200, random_state=0):
        self.n_electrons = n_electrons
        self.rng = np.random.RandomState(random_state)
        # Initial Guess
        phi = self.rng.uniform(0, 2 * np.pi, self.n_electrons)
        theta = self.rng.uniform(-np.pi, np.pi, self.n_electrons)
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        self.x0 = np.hstack((x, y, z))
        # Initial Multiplier
        self.v0 = np.zeros(self.n_electrons)

    def _get_cordinates(self, x):
        x_coord = x[:self.n_electrons]
        y_coord = x[self.n_electrons:2 * self.n_electrons]
        z_coord = x[2 * self.n_electrons:]
        return x_coord, y_coord, z_coord

    def _compute_coordinate_deltas(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        dx = x_coord[:, None] - x_coord
        dy = y_coord[:, None] - y_coord
        dz = z_coord[:, None] - z_coord
        return dx, dy, dz

    def fun(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        with np.errstate(divide='ignore'):
            dm1 = (dx**2 + dy**2 + dz**2) ** -0.5
        dm1[np.diag_indices_from(dm1)] = 0
        return 0.5 * np.sum(dm1)

    def grad(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)

        with np.errstate(divide='ignore'):
            dm3 = (dx**2 + dy**2 + dz**2) ** -1.5
        dm3[np.diag_indices_from(dm3)] = 0

        grad_x = -np.sum(dx * dm3, axis=1)
        grad_y = -np.sum(dy * dm3, axis=1)
        grad_z = -np.sum(dz * dm3, axis=1)

        return np.hstack((grad_x, grad_y, grad_z))

    def hess(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        d = (dx**2 + dy**2 + dz**2) ** 0.5

        with np.errstate(divide='ignore'):
            dm3 = d ** -3
            dm5 = d ** -5

        i = np.arange(self.n_electrons)
        dm3[i, i] = 0
        dm5[i, i] = 0

        Hxx = dm3 - 3 * dx**2 * dm5
        Hxx[i, i] = -np.sum(Hxx, axis=1)

        Hxy = -3 * dx * dy * dm5
        Hxy[i, i] = -np.sum(Hxy, axis=1)

        Hxz = -3 * dx * dz * dm5
        Hxz[i, i] = -np.sum(Hxz, axis=1)

        Hyy = dm3 - 3 * dy**2 * dm5
        Hyy[i, i] = -np.sum(Hyy, axis=1)

        Hyz = -3 * dy * dz * dm5
        Hyz[i, i] = -np.sum(Hyz, axis=1)

        Hzz = dm3 - 3 * dz**2 * dm5
        Hzz[i, i] = -np.sum(Hzz, axis=1)

        H = np.vstack((
            np.hstack((Hxx, Hxy, Hxz)),
            np.hstack((Hxy, Hyy, Hyz)),
            np.hstack((Hxz, Hyz, Hzz))
        ))

        return H

    def constr_eq(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        return x_coord**2 + y_coord**2 + z_coord**2 - 1

    def jac_eq(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        Jx = 2 * np.diag(x_coord)
        Jy = 2 * np.diag(y_coord)
        Jz = 2 * np.diag(z_coord)

        return csc_matrix(np.hstack((Jx, Jy, Jz)))

    def hess_eq(self, x, v):
        D = 2 * np.diag(v)
        return block_diag(D, D, D)


class Rosenbrock(MatlabLikeInterface):
    """Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    """

    def __init__(self, n=2, random_state=0):
        rng = np.random.RandomState(random_state)
        self.x0 = rng.uniform(-1, 1, n)
        self.v0 = []
        self.x_opt = np.ones(n)

    def fun(self, x):
        x = np.asarray(x)
        r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                   axis=0)
        return r

    def grad(self, x):
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (200 * (xm - xm_m1**2) -
                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2]**2)
        return der

    def hess(self, x):
        x = np.atleast_1d(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H


class BoundContrRosenbrock(Rosenbrock):
    """Bound constrained Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
        subject to: -1 <= x <=0
    """

    def __init__(self, n=2, random_state=0):
        Rosenbrock.__init__(self, n, random_state)
        self.x_opt = np.zeros(n)
        self.lb = -1*np.ones(n)
        self.ub = np.zeros(n)


class IneqLinearConstrRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1

    Taken from matlab ``fmincon`` documentation.
    """

    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.v0 = [0]
        self.x0 = [-1, -0.5]
        self.A_ineq = np.array([1, 2])
        self.b_ineq = np.array([1])
        self.x_opt = [0.5022, 0.2489]


class LinearConstrRosenbrock(Rosenbrock):
    """Rosenbrock subject to equality and inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1
                    2 x[0] + x[1] = 1

    Taken from matlab ``fimincon`` documentation.
    """

    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.v0 = [0]
        self.x0 = [-1, -0.5]
        self.A_ineq = np.array([1, 2])
        self.b_ineq = np.array([1])
        self.A_eq = np.array([2, 1])
        self.b_eq = np.array([1])
        self.x_opt = [0.41494,  0.17011]

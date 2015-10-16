#!/usr/bin/env python

"""
This file tests basic operations on **all** types supported by QR_MUMPS
and on symmetric and general matrices.
"""

from qr_mumps.solver import QRMUMPSSolver
import numpy as np
from numpy.testing import *
import sys


{% for index_type in index_list %}
    {% for element_type in type_list %}
class NumpyQRMUMPSSolverTestCaseMoreLinesThanColumns_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        self.A = np.array([[0,0.1,0,0,0], [0.7,0,0.3,0.5,0.1], [0.6,0,0,0.2,0], [0,0,0.6,0,0.6], [0,0,0.7,0,0], [0.4,0.1,0,0,0], [0,0,0.2,0,0]], dtype=np.@element_type|lower@)
        self.arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.@index_type|lower@)
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.@index_type|lower@)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.@element_type|lower@)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert_equal(self.n, solver.n)

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert(solver.analyzed==True)

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert(solver.analyzed==True)
        assert(solver.factorized==True)

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)


class NumpyQRMUMPSSolverTestCaseMoreColumnsThanLines_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        self.A = np.array([[0,0.7,0.6,0,0,0.4,0,], [0.1,0,0,0,0,0.1,0], [0,0.3,0,0.6,0.7,0,0.2], [0,0.5,0.2,0,0,0,0], [0,0.1,0,0.6,0,0,0]], dtype=np.@element_type|lower@)
        self.acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.@index_type|lower@)
        self.arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.@index_type|lower@)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.@element_type|lower@)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert_equal(self.n, solver.n)
        assert_equal(self.m, solver.m)

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert(solver.analyzed==True)

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert(solver.analyzed==True)
        assert(solver.factorized==True)

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert_almost_equal(np.dot(self.A,x), rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        print x
        assert_almost_equal(np.dot(self.A,x), rhs, 5)


  {% endfor %}
{% endfor %}

if __name__ == "__main__":
      run_module_suite()

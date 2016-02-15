#!/usr/bin/env python

"""
This file tests basic operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""
cysparse_installed = False
try:
    from cysparse.sparse.ll_mat import *
    import cysparse.common_types.cysparse_types as types
    cysparse_installed = True
except:
    print "CySparse is not installed, no tests will be run."
    pass


from qr_mumps.solver import QRMUMPSSolver
import numpy as np
from numpy.testing import *
import sys


class CySparseQRMUMPSSolverInput(TestCase):
    def setUp(self):
        pass

    def test_index(self):
        m = 7
        n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int64)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int64)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)
        A = LLSparseMatrix(nrow=m, ncol=n, itype=types.INT64_T, dtype=types.FLOAT64_T)
        A.put_triplet(arow, acol, aval)
        assert_raises(TypeError, QRMUMPSSolver, A)

    def test_type(self):
        m = 7
        n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float128)
        A = LLSparseMatrix(nrow=m, ncol=n, itype=types.INT32_T, dtype=types.FLOAT128_T)
        A.put_triplet(arow, acol, aval)
        assert_raises(TypeError, QRMUMPSSolver, A)


{% for index_type in index_list %}
    {% for element_type in type_list %}
class CySparseQRMUMPSSolverTestCaseMoreLinesThanColumns_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.@index_type|lower@)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.@index_type|lower@)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.@element_type|lower@)
        self.A = LLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.@index_type@_T, dtype=types.@element_type@_T)
        self.A.put_triplet(arow, acol, aval)
        print self.A

    def test_init(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        assert_equal(self.n, solver.n)

    def test_analyze(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.analyze()
        assert(solver.analyzed==True)

    def test_factorize(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        assert(solver.analyzed==True)
        assert(solver.factorized==True)

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.m, 3], dtype=np.@element_type|lower@)
        rhs[:, 0] = self.A*B[:, 0]
        rhs[:, 1] = self.A*B[:, 1]
        rhs[:, 2] = self.A*B[:, 2]
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.m, 3], dtype=np.@element_type|lower@)
        rhs[:, 0] = self.A*B[:, 0]
        rhs[:, 1] = self.A*B[:, 1]
        rhs[:, 2] = self.A*B[:, 2]
        x = solver.least_squares(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.@element_type|lower@)
        assert_raises(ValueError, solver.least_squares, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.m, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        assert_raises(RuntimeError, solver.minimum_norm, B)
  

class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.@index_type|lower@)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.@index_type|lower@)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.@element_type|lower@)
        self.A = LLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.@index_type@_T, dtype=types.@element_type@_T)
        self.A.put_triplet(arow, acol, aval)

    def test_init(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        assert_equal(self.m, solver.m)
        assert_equal(self.n, solver.n)

    def test_analyze(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.analyze()
        assert(solver.analyzed==True)

    def test_factorize(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        assert(solver.analyzed==True)
        assert(solver.factorized==True)

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.m, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        x = solver.solve(B)
        assert_almost_equal(self.A*x[:, 0], B[:, 0], 5)
        assert_almost_equal(self.A*x[:, 1], B[:, 1], 5)
        assert_almost_equal(self.A*x[:, 2], B[:, 2], 5)

    def test_least_squares(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.@element_type|lower@)
        assert_raises(RuntimeError, solver.least_squares, rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.@element_type|lower@)
        assert_raises(ValueError, solver.minimum_norm, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.m, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        x = solver.minimum_norm(B)
        assert_almost_equal(self.A*x[:, 0], B[:, 0], 5)
        assert_almost_equal(self.A*x[:, 1], B[:, 1], 5)
        assert_almost_equal(self.A*x[:, 2], B[:, 2], 5)


  {% endfor %}
{% endfor %}
           
if __name__ == "__main__":
      run_module_suite()


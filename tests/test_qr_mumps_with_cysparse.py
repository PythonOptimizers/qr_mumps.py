#!/usr/bin/env python

"""
This file tests basic operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""
from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

from qr_mumps.solver import QRMUMPSSolver
import numpy as np
from numpy.testing import *
import sys


class CySparseQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float32)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.FLOAT32_T)
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
        e = np.ones(self.n, dtype=np.float32)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.least_squares(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float32)
        assert_raises(ValueError, solver.least_squares, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        assert_raises(RuntimeError, solver.minimum_norm, rhs)
  

class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float32)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.FLOAT32_T)
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
        e = np.ones(self.n, dtype=np.float32)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_least_squares(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.float32)
        assert_raises(RuntimeError, solver.least_squares, rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float32)
        assert_raises(ValueError, solver.minimum_norm, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.minimum_norm(rhs)
        assert_almost_equal(self.A*x, rhs, 5)



class CySparseQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_FLOAT64(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.FLOAT64_T)
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
        e = np.ones(self.n, dtype=np.float64)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.least_squares(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float64)
        assert_raises(ValueError, solver.least_squares, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        assert_raises(RuntimeError, solver.minimum_norm, rhs)
  

class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_FLOAT64(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.FLOAT64_T)
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
        e = np.ones(self.n, dtype=np.float64)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_least_squares(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.float64)
        assert_raises(RuntimeError, solver.least_squares, rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float64)
        assert_raises(ValueError, solver.minimum_norm, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.minimum_norm(rhs)
        assert_almost_equal(self.A*x, rhs, 5)



class CySparseQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_COMPLEX64(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.COMPLEX64_T)
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
        e = np.ones(self.n, dtype=np.complex64)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.least_squares(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex64)
        assert_raises(ValueError, solver.least_squares, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        assert_raises(RuntimeError, solver.minimum_norm, rhs)
  

class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_COMPLEX64(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.COMPLEX64_T)
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
        e = np.ones(self.n, dtype=np.complex64)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_least_squares(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.complex64)
        assert_raises(RuntimeError, solver.least_squares, rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex64)
        assert_raises(ValueError, solver.minimum_norm, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.minimum_norm(rhs)
        assert_almost_equal(self.A*x, rhs, 5)



class CySparseQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_COMPLEX128(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex128)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.COMPLEX128_T)
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
        e = np.ones(self.n, dtype=np.complex128)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(x, e, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.least_squares(rhs)
        assert_almost_equal(x, B, 5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex128)
        assert_raises(ValueError, solver.least_squares, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        assert_raises(RuntimeError, solver.minimum_norm, rhs)
  

class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_COMPLEX128(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex128)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.COMPLEX128_T)
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
        e = np.ones(self.n, dtype=np.complex128)
        rhs = self.A * e
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.solve(rhs)
        assert_almost_equal(self.A*x, rhs, 5)

    def test_least_squares(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.complex128)
        assert_raises(RuntimeError, solver.least_squares, rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex128)
        assert_raises(ValueError, solver.minimum_norm, rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.minimum_norm(rhs)
        assert_almost_equal(self.A*x, rhs, 5)



           
if __name__ == "__main__":
      run_module_suite()

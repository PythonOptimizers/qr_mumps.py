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


class CySparseQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float32)
        self.A = NewLLSparseMatrix(nrow=self.m, ncol=self.n, itype=types.INT32_T, dtype=types.FLOAT32_T)
        self.A.put_triplet(arow, acol, aval)
        print self.A

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
        print 'rhs:', rhs
        print 'x:', x
        print 'Ax:', self.A*x
        assert_almost_equal(self.A*x, rhs, 5)

#    def test_dense_solve_multiple_rhs(self):
#        print 'allo'
#        solver = QRMUMPSSolver(self.A, verbose=False)
#        B = np.ones([self.n, 3], dtype=np.float32)
#        B[: ,1] = 2 * B[:,1]
#        B[: ,2] = 3 * B[:,2]
#        rhs = self.A * B
#        print rhs
#        x = solver.solve(rhs)
#        print x
#        assert_almost_equal(self.A*x, rhs, 5)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver(self.A, verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = self.A * B
        x = solver.minimum_norm(rhs)
        print x
        assert_almost_equal(self.A*x, rhs, 5)
        del solver

#    def test_dense_minimum_norm_multiple_rhs1(self):
#        solver1 = QRMUMPSSolver(self.A, verbose=False)
#        B = np.ones([self.n, 3], dtype=np.float32)
#        B[: ,1] = 2 * B[:,1]
#        B[: ,2] = 3 * B[:,2]
#        rhs = self.A * B
#        x = solver1.minimum_norm(rhs)
#        print x
#        assert_almost_equal(self.A*x, rhs, 5)


if __name__ == "__main__":
      run_module_suite()

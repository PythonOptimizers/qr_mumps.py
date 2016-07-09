#!/usr/bin/env python

"""
This file tests basic operations on **all** types supported by QR_MUMPS
and on symmetric and general matrices.
"""

from qr_mumps.solver import QRMUMPSSolver
import numpy as np
import pytest
from unittest import TestCase
import sys


class NumpyQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_COMPLEX64(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        self.A = np.array([[0,0.1,0,0,0], [0.7,0,0.3,0.5,0.1], [0.6,0,0,0.2,0], [0,0,0.6,0,0.6], [0,0,0.7,0,0], [0.4,0.1,0,0,0], [0,0,0.2,0,0]], dtype=np.complex64)
        self.arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n

    def test_number_of_args(self):
	with pytest.raises(ValueError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval, 1))
	with pytest.raises(TypeError):
            solver = QRMUMPSSolver(self.m)

    def test_entry_index_types(self):
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.float128)
        with pytest.raises(TypeError):
	    solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_entry_element_types(self):
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float128)
        with pytest.raises(TypeError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed is True
        assert solver.factorized is True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(x, e, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = solver.least_squares(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex64)
	with pytest.raises(ValueError):
            x = solver.least_squares(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
	with pytest.raises(RuntimeError):
            x = solver.minimum_norm(rhs)


class NumpyQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_COMPLEX64(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        self.A = np.array([[0,0.7,0.6,0,0,0.4,0,], [0.1,0,0,0,0,0.1,0], [0,0.3,0,0.6,0.7,0,0.2], [0,0.5,0.2,0,0,0,0], [0,0.1,0,0.6,0,0,0]], dtype=np.complex64)
        self.acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n
        assert self.m == solver.m

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed==True
        assert solver.factorized==True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_least_squares(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.complex64)
	with pytest.raises(RuntimeError):
            x = solver.least_squares(rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex64)
	with pytest.raises(ValueError):
            x = solver.minimum_norm(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
        x = solver.minimum_norm(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)



class NumpyQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_COMPLEX128(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        self.A = np.array([[0,0.1,0,0,0], [0.7,0,0.3,0.5,0.1], [0.6,0,0,0.2,0], [0,0,0.6,0,0.6], [0,0,0.7,0,0], [0.4,0.1,0,0,0], [0,0,0.2,0,0]], dtype=np.complex128)
        self.arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex128)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n

    def test_number_of_args(self):
	with pytest.raises(ValueError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval, 1))
	with pytest.raises(TypeError):
            solver = QRMUMPSSolver(self.m)

    def test_entry_index_types(self):
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.float128)
        with pytest.raises(TypeError):
	    solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_entry_element_types(self):
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float128)
        with pytest.raises(TypeError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed is True
        assert solver.factorized is True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(x, e, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = solver.least_squares(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex128)
	with pytest.raises(ValueError):
            x = solver.least_squares(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
	with pytest.raises(RuntimeError):
            x = solver.minimum_norm(rhs)


class NumpyQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_COMPLEX128(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        self.A = np.array([[0,0.7,0.6,0,0,0.4,0,], [0.1,0,0,0,0,0.1,0], [0,0.3,0,0.6,0.7,0,0.2], [0,0.5,0.2,0,0,0,0], [0,0.1,0,0.6,0,0,0]], dtype=np.complex128)
        self.acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex128)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n
        assert self.m == solver.m

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed==True
        assert solver.factorized==True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_least_squares(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.complex128)
	with pytest.raises(RuntimeError):
            x = solver.least_squares(rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.complex128)
	with pytest.raises(ValueError):
            x = solver.minimum_norm(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
        x = solver.minimum_norm(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)



class NumpyQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        self.A = np.array([[0,0.1,0,0,0], [0.7,0,0.3,0.5,0.1], [0.6,0,0,0.2,0], [0,0,0.6,0,0.6], [0,0,0.7,0,0], [0.4,0.1,0,0,0], [0,0,0.2,0,0]], dtype=np.float32)
        self.arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float32)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n

    def test_number_of_args(self):
	with pytest.raises(ValueError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval, 1))
	with pytest.raises(TypeError):
            solver = QRMUMPSSolver(self.m)

    def test_entry_index_types(self):
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.float128)
        with pytest.raises(TypeError):
	    solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_entry_element_types(self):
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float128)
        with pytest.raises(TypeError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed is True
        assert solver.factorized is True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(x, e, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = solver.least_squares(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float32)
	with pytest.raises(ValueError):
            x = solver.least_squares(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
	with pytest.raises(RuntimeError):
            x = solver.minimum_norm(rhs)


class NumpyQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        self.A = np.array([[0,0.7,0.6,0,0,0.4,0,], [0.1,0,0,0,0,0.1,0], [0,0.3,0,0.6,0.7,0,0.2], [0,0.5,0.2,0,0,0,0], [0,0.1,0,0.6,0,0,0]], dtype=np.float32)
        self.acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float32)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n
        assert self.m == solver.m

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed==True
        assert solver.factorized==True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_least_squares(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.float32)
	with pytest.raises(RuntimeError):
            x = solver.least_squares(rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float32)
	with pytest.raises(ValueError):
            x = solver.minimum_norm(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float32)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
        x = solver.minimum_norm(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)



class NumpyQRMUMPSSolverTestCaseMoreLinesThanColumns_INT32_FLOAT64(TestCase):
    def setUp(self):
        self.m = 7
        self.n = 5
        self.A = np.array([[0,0.1,0,0,0], [0.7,0,0.3,0.5,0.1], [0.6,0,0,0.2,0], [0,0,0.6,0,0.6], [0,0,0.7,0,0], [0.4,0.1,0,0,0], [0,0,0.2,0,0]], dtype=np.float64)
        self.arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n

    def test_number_of_args(self):
	with pytest.raises(ValueError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval, 1))
	with pytest.raises(TypeError):
            solver = QRMUMPSSolver(self.m)

    def test_entry_index_types(self):
        self.acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.float128)
        with pytest.raises(TypeError):
	    solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_entry_element_types(self):
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float128)
        with pytest.raises(TypeError):
            solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval))

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed is True
        assert solver.factorized is True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(x, e, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = solver.least_squares(rhs)
        assert np.allclose(x, B, 1e-5)

    def test_dense_least_squares_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float64)
	with pytest.raises(ValueError):
            x = solver.least_squares(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
	with pytest.raises(RuntimeError):
            x = solver.minimum_norm(rhs)


class NumpyQRMUMPSSolverTestCaseMoreColumnsThanLines_INT32_FLOAT64(TestCase):
    def setUp(self):
        self.m = 5
        self.n = 7
        self.A = np.array([[0,0.7,0.6,0,0,0.4,0,], [0.1,0,0,0,0,0.1,0], [0,0.3,0,0.6,0.7,0,0.2], [0,0.5,0.2,0,0,0,0], [0,0.1,0,0.6,0,0,0]], dtype=np.float64)
        self.acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
        self.arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
        self.aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)

    def test_init(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        assert self.n == solver.n
        assert self.m == solver.m

    def test_analyze(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.analyze()
        assert solver.analyzed==True

    def test_factorize(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        assert solver.analyzed==True
        assert solver.factorized==True

    def test_dense_solve_single_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = np.dot(self.A, e)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_dense_solve_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        solver.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = solver.solve(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)

    def test_least_squares(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m, 1], dtype=np.float64)
	with pytest.raises(RuntimeError):
            x = solver.least_squares(rhs)
          
    def test_dense_minimum_norm_wrong_size_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        rhs = np.ones([self.m+1, 1], dtype=np.float64)
	with pytest.raises(ValueError):
            x = solver.minimum_norm(rhs)

    def test_dense_minimum_norm_multiple_rhs(self):
        solver = QRMUMPSSolver((self.m, self.n, self.arow, self.acol, self.aval), verbose=False)
        B = np.ones([self.n, 3], dtype=np.float64)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A, B)
        x = solver.minimum_norm(rhs)
        assert np.allclose(np.dot(self.A,x), rhs, 1e-5)




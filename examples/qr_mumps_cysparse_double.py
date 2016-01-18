from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types

from qr_mumps.solver import QRMUMPSSolver

import numpy as np

import sys

# Read Matrix in Matrix Market format from command line argument
A = LLSparseMatrix(mm_filename=sys.argv[1],
                   itype=types.INT32_T,
                   dtype=types.FLOAT64_T)
(m, n) = A.shape

print "A:"
print A


e = np.ones(n, 'd')
rhs = A*e

print "rhs:"
print rhs

solver = QRMUMPSSolver(A, verbose=True)

solver.analyze()

solver.factorize()

x = solver.least_squares(rhs)
np.testing.assert_almost_equal(A*x, rhs, 5)

x = solver.solve(rhs)
np.testing.assert_almost_equal(x, e)

print "= " * 80

rhs = np.ones([n, 3], "d")
rhs[:, 1] = 2 * rhs[:, 1]
rhs[:, 2] = 3 * rhs[:, 2]
print "rhs:"
print rhs

x = solver.solve(rhs)
print "x:"
print x

np.testing.assert_almost_equal(A*x[:, 0], rhs[:, 0])
np.testing.assert_almost_equal(A*x[:, 1], rhs[:, 1])
np.testing.assert_almost_equal(A*x[:, 2], rhs[:, 2])

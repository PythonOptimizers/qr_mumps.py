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


solver = QRMUMPSSolver(A, verbose=True)
print solver
solver.analyze()

solver.factorize()

print "\n\nFinding a least-square solution to Ax = b where b is a vector"
e = np.ones(n, 'd')
b = A*e
print "b:"
print b

x = solver.least_squares(b)
np.testing.assert_almost_equal(A*x, b, 5)
print "x:"
print x

print "\n"+"=" * 80+"\n"
print "Solving Ax = B where B is a matrix"

B = np.ones([m, 3], "d")
B[:, 1] = 2 * B[:, 1]
B[:, 2] = 3 * B[:, 2]
print "B:"
print B

x = solver.solve(B)
print "x:"
print x

np.testing.assert_almost_equal(A*x[:, 0], B[:, 0])
np.testing.assert_almost_equal(A*x[:, 1], B[:, 1])
np.testing.assert_almost_equal(A*x[:, 2], B[:, 2])

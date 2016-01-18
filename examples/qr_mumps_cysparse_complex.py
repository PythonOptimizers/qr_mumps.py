from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
from qr_mumps.solver import QRMUMPSSolver

import numpy as np

import sys


A = LLSparseMatrix(mm_filename=sys.argv[1], itype=types.INT32_T, dtype=types.COMPLEX128_T)
(n, m) = A.shape
print "A:"
print A

e = np.ones(n, dtype=np.complex128)
rhs = A*e

solver = QRMUMPSSolver(A, verbose=True)

solver.analyze()

solver.factorize()

x = solver.solve(rhs)
print "x should be 1-column vector:"
print x

print "=" * 80

rhs = np.ones([n, 3], dtype=np.complex128)
rhs[:, 1] = 2 * rhs[:, 1]
rhs[:, 2] = 3 * rhs[:, 2]

x = solver.solve(rhs)
print "x:"
print x

print x[:,0]

print "x" * 80

x = solver.solve(A.to_ndarray())
print "x should be the identity matrix of size %d"%m
print x

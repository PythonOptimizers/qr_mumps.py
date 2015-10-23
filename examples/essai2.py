from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

from qr_mumps.solver import QRMUMPSSolver

import numpy as np

import sys


A = NewLLSparseMatrix(mm_filename=sys.argv[1], itype=types.INT32_T, dtype=types.FLOAT64_T)
(m, n) = A.shape
(i,j,val) = A.find()
(i1,j1,val1) = A.triu(1).find()
row = np.hstack((i,i1))
col = np.hstack((j,j1))
val = np.hstack((val, val1))

B = NewLLSparseMatrix(nrow=m, ncol=n, itype=types.INT32_T, dtype=types.FLOAT64_T)
B.put_triplet(row, col, val)


e = np.ones(n, 'd')
rhs = A*e
for i in xrange(0,10):
    solver = QRMUMPSSolver(B, verbose=True)


solver.analyze()

solver.factorize()
x = solver.least_squares(rhs)
print rhs
np.testing.assert_almost_equal(x,e)

x = solver.solve(rhs)
np.testing.assert_almost_equal(x,e)

print "= " * 80

B = np.ones([n, 3], "d")
B[: ,1] = 2 * B[:,1]
B[: ,2] = 3 * B[:,2]

rhs = A * B

x = solver.solve(rhs)
np.testing.assert_almost_equal(x,B)

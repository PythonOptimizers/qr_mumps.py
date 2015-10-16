from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

from qr_mumps.solver import MUMPSSolver

import numpy as np

import sys


A = NewLLSparseMatrix(mm_filename=sys.argv[1], itype=types.INT32_T, dtype=types.FLOAT64_T)

print A


(n, m) = A.shape
e = np.ones(n, 'd')
rhs = A*e

solver = MUMPSSolver(A, verbose=True)

solver.analyze()

solver.factorize()

x = solver.solve(rhs)

x = solver.refine(rhs, 10)

print x
sys.exit(0)


print "= " * 80

B = np.ones([n, 3], "d")
B[: ,1] = 2 * B[:,1]
B[: ,2] = 3 * B[:,2]

rhs = A * B

x = solver.solve(rhs)
print x


from qr_mumps.qr_mumps_context import qr_mumpsContext
import numpy as np
import sys

m = 4
n = 4
A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.float64)
arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float64)

context = qr_mumpsContext((m, n, arow, acol, aval), verbose=False)
print "ca marche"
context.analyze()
print context.analyzed

context.factorize()

e = np.ones(n, dtype=np.float64)
rhs = np.dot(A, e)

x = context.solve(rhs)
print rhs, x
np.testing.assert_almost_equal(x,e)


print "= " * 80

B = np.ones([n, 3], dtype=np.float64)
B[: ,1] = 2 * B[:,1]
B[: ,2] = 3 * B[:,2]
rhs = np.dot(A,B)

x = context.solve(rhs)
np.testing.assert_almost_equal(x,B)


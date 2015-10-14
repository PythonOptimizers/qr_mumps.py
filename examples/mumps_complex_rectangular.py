from qr_mumps.qr_mumps_context import qr_mumpsContext
import numpy as np
import sys

m = 7
n = 5
#A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.float64)
arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)

context = qr_mumpsContext((m, n, arow, acol, aval), verbose=False)
print "ca marche"
context.analyze()
print context.analyzed

context.factorize()

rhs = np.ones(n, dtype=np.complex64)
#rhs = np.dot(A, e)

x = context.solve(rhs)
print rhs, x

from qr_mumps.solver import QRMUMPSSolver
import numpy as np

m = 4
n = 4
A = np.array([[1, 2, 3, 4],
              [5, 0, 7, 8],
              [9, 10, 0, 12],
              [13, 14, 15, 0]],
             dtype=np.float64)
arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                dtype=np.int32)
acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                dtype=np.int32)
aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0],
                dtype=np.float64)


print "A:"
print A

solver = QRMUMPSSolver((m, n, arow, acol, aval), verbose=False)

solver.analyze()
print solver.analyzed

solver.factorize()

e = np.ones(n, dtype=np.float64)
rhs = np.dot(A, e)
print "rhs:"
print rhs

x = solver.solve(rhs)
np.testing.assert_almost_equal(x, e)
print "x:"
print x


print "=" * 80

B = np.ones([n, 3], dtype=np.float64)
B[:, 1] = 2 * B[:, 1]
B[:, 2] = 3 * B[:, 2]
rhs = np.dot(A, B)
print "rhs:"
print rhs

x = solver.solve(rhs)
np.testing.assert_almost_equal(x, B)
print "x:"
print x

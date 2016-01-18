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

solver.factorize()

print "\n\nSolving Ax = b where b is a vector"
e = np.ones(n, dtype=np.float64)
b = np.dot(A, e)
print "b:"
print b

x = solver.solve(b)
np.testing.assert_almost_equal(x, e)
print "x: it should be a vector of ones"
print x


print "\n"+"=" * 80+"\n"
print "Solving Ax = B where B is a matrix"

E = np.ones([n, 3], dtype=np.float64)
E[:, 1] = 2 * E[:, 1]
E[:, 2] = 3 * E[:, 2]
B = np.dot(A, E)
print "B:"
print B

x = solver.solve(B)
np.testing.assert_almost_equal(x, E)
print "x: it should be the m x 3 matrix "
print "   with 1 on the first column, 2 on the second and 3 on the third."
print x

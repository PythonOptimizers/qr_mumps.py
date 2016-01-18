from qr_mumps.solver import QRMUMPSSolver

import numpy as np

m = 5
n = 7
A = np.array([[0, 0.1, 0, 0, 0],
              [0.7, 0, 0.3, 0.5, 0.1],
              [0.6, 0, 0, 0.2, 0],
              [0, 0, 0.6, 0, 0.6],
              [0, 0, 0.7, 0, 0],
              [0.4, 0.1, 0, 0, 0],
              [0, 0, 0.2, 0, 0]], dtype=np.float64)
A = A.T
arow = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
acol = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.float64)
print "A:"
print A

print "-"*80
print "Loop intantiating a solver and solving with one rhs"
print "-"*80


for i in xrange(0,10):
    print "    "+"-"*76
    print "    i = %d"%i
    print "    "+"-"*76
    solver = QRMUMPSSolver((m, n, arow, acol, aval), verbose=False)

    solver.factorize('scotch')
    print solver.factorization_statistics

    e = (i+1)*np.ones(n, dtype=np.float64)
    rhs = np.dot(A, e)
    print "    rhs:"
    print rhs

    x = solver.solve(rhs)
    print "    x:"
    print x



print "-"*80
print "solving with multiple rhs"
print "-"*80

B = np.ones([n, 3], dtype=np.float64)
B[:, 1] = 2 * B[:, 1]
B[:, 2] = 3 * B[:, 2]
rhs = np.dot(A, B)
print "rhs:"
print rhs

x = solver.solve(rhs)
print "x:"
print x

print "Ax:"
print np.dot(A, x)

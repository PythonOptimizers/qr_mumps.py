from qr_mumps.solver import QRMUMPSSolver
import numpy as np

m = 7
n = 5
arow = np.array([1,2,5,0,5,1,3,4,6,1,2,1,3], dtype=np.int32)
acol = np.array([0,0,0,1,1,2,2,2,2,3,3,4,4], dtype=np.int32)
aval = np.array([0.7,0.6,0.4,0.1,0.1,0.3,0.6,0.7,0.2,0.5,0.2,0.1,0.6], dtype=np.complex64)

solver = QRMUMPSSolver((m, n, arow, acol, aval), verbose=False)
solver.analyze()

solver.factorize()

rhs = np.ones(m, dtype=np.complex64)
print "rhs:"
print rhs

x = solver.solve(rhs)
print "x:"
print x

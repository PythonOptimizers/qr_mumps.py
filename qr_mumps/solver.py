"""
Factory method to access qr_mumps.
"""
import numpy as np

from qr_mumps.src.numpy_qr_mumps_INT32_COMPLEX64 import NumpyQRMUMPSSolver_INT32_COMPLEX64
from qr_mumps.src.numpy_qr_mumps_INT32_COMPLEX128 import NumpyQRMUMPSSolver_INT32_COMPLEX128
from qr_mumps.src.numpy_qr_mumps_INT32_FLOAT32 import NumpyQRMUMPSSolver_INT32_FLOAT32
from qr_mumps.src.numpy_qr_mumps_INT32_FLOAT64 import NumpyQRMUMPSSolver_INT32_FLOAT64

cysparse_installed = False
try:
    from qr_mumps.src.cysparse_qr_mumps_INT32_COMPLEX64 import CySparseQRMUMPSSolver_INT32_COMPLEX64
    from qr_mumps.src.cysparse_qr_mumps_INT32_COMPLEX128 import CySparseQRMUMPSSolver_INT32_COMPLEX128
    from qr_mumps.src.cysparse_qr_mumps_INT32_FLOAT32 import CySparseQRMUMPSSolver_INT32_FLOAT32
    from qr_mumps.src.cysparse_qr_mumps_INT32_FLOAT64 import CySparseQRMUMPSSolver_INT32_FLOAT64
    from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
    from cysparse.common_types.cysparse_types import *
    cysparse_installed = True
except:
    pass

allowed_types = '\titype:INT32\n\tdtype:COMPLEX64,COMPLEX128,FLOAT32,FLOAT64\n'
type_error_msg = 'Matrix has an index and/or element type that is incompatible \n'
type_error_msg += 'with qr_mumps\nAllowed types:\n%s' % allowed_types

def QRMUMPSSolver(arg1, verbose=False):
    """
    Create and return the right qr_mumps solver based on the element type
    supplied as input.

    qr_mumps ("MUltifrontal Massively Parallel Solver") is a package for solving
    systems of linear equations of the form Ax = b,
    where A is a square **sparse** matrix that can be either unsymmetric,
    symmetric positive definite, or general symmetric, on distributed memory
    computers.

    qr_mumps performs a Gaussian factorization
      A = LU
    where L is a lower triangular matrix and U an upper triangular matrix.

    If the matrix is symmetric then qr_mumps performs the factorization
      A = LDL^T
    where D is block diagonal matrix.

    Args:
        m: number of line of matrix A
        n: number of column of matrix A
        a_row: row indices of non zero elements of A
        a_col: column indices of non zero elements of A
        a_val: values of non zeros elements of A
        verbose: a boolean to turn on or off the verbosity of qr_mumps
    """
    if isinstance(arg1, tuple):
        if len(arg1) != 5:
            raise ValueError("If a tuple is supplied, it must have 5"+
                             "items: m, n, a_row, a_col, a_val")
        m = arg1[0]
        n = arg1[1]
        a_row = arg1[2]
        a_col = arg1[3]
        a_val = arg1[4]

        itype = a_row.dtype
        dtype = a_val.dtype

        if itype != a_col.dtype:
            raise TypeError(type_error_msg)

        if itype == np.int32:
            if dtype == np.complex64:
                solver = NumpyQRMUMPSSolver_INT32_COMPLEX64(m, n, a_row.size, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.complex128:
                solver = NumpyQRMUMPSSolver_INT32_COMPLEX128(m, n, a_row.size, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.float32:
                solver = NumpyQRMUMPSSolver_INT32_FLOAT32(m, n, a_row.size, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.float64:
                solver = NumpyQRMUMPSSolver_INT32_FLOAT64(m, n, a_row.size, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            else:
                raise TypeError(type_error_msg)
        else:
            raise TypeError(type_error_msg)


    elif cysparse_installed:
        if not PyLLSparseMatrix_Check(arg1):
            raise TypeError('arg1 should be a LLSparseMatrix')

        A = arg1
        itype = A.itype
        dtype = A.dtype
        m = A.nrow
        n = A.ncol

        if itype == INT32_T:
            if dtype == COMPLEX64_T:
                solver = CySparseQRMUMPSSolver_INT32_COMPLEX64(m, n, A.nnz, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == COMPLEX128_T:
                solver = CySparseQRMUMPSSolver_INT32_COMPLEX128(m, n, A.nnz, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == FLOAT32_T:
                solver = CySparseQRMUMPSSolver_INT32_FLOAT32(m, n, A.nnz, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == FLOAT64_T:
                solver = CySparseQRMUMPSSolver_INT32_FLOAT64(m, n, A.nnz, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            else:
                raise TypeError(type_error_msg)

        else:
            raise TypeError(type_error_msg)
    else:
        raise TypeError("This matrix type is not recognized/implemented")

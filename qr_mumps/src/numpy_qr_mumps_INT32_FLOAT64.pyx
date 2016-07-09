from qr_mumps.src.qr_mumps_INT32_FLOAT64 cimport BaseQRMUMPSSolver_INT32_FLOAT64, c_to_fortran_index_array

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy

cimport numpy as cnp
cnp.import_array()

cdef class NumpyQRMUMPSSolver_INT32_FLOAT64(BaseQRMUMPSSolver_INT32_FLOAT64):
    """
    QR_MUMPS solver when matrix A is supplied in coordinate format through Numpy arrays.

    Args:
        m: number of lines of matrix A
        n: number of columns of matrix A
        a_row: row indices of non zero elements of A
        a_col: column indices of non zero elements of A
        a_val: values of non zeros elements of A
        verbose: a boolean to turn on or off the verbosity of MUMPS

    Warning: if the numpy arrays are modified externally by the user between
    two calls to solve, the changes in arrays won't be passed to QR_MUMPS.
    """
    def __cinit__(self, int m, int n, int nnz, verbose=False):
        pass

    def __dealloc__(self):
        PyMem_Free(self.params.irn)
        PyMem_Free(self.params.jcn)

    cpdef get_matrix_data(self, cnp.ndarray[cnp.int32_t, ndim=1] arow,
                                cnp.ndarray[cnp.int32_t, ndim=1] acol,
                                cnp.ndarray[cnp.float64_t, ndim=1] aval):
        """
        Args:
            arow: row indices of non zero elements of A
            acol: column indices of non zero elements of A
            aval: values of non zeros elements of A

        Note: we keep the same name for this method in all derived classes.
        """
        # allocate memory for irn and jcn
        self.params.irn = <int *> PyMem_Malloc(self.nnz * sizeof(int))
        self.params.jcn = <int *> PyMem_Malloc(self.nnz * sizeof(int))

        memcpy(self.params.irn, <int *> cnp.PyArray_DATA(arow), self.nnz*sizeof(int))
        memcpy(self.params.jcn, <int *> cnp.PyArray_DATA(acol), self.nnz*sizeof(int))

        self.params.val = <double *> cnp.PyArray_DATA(aval)

        # convert irn and jcn indices to Fortran format
        self.index_to_fortran()

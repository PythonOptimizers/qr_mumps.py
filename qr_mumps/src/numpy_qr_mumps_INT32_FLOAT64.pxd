from qr_mumps.src.qr_mumps_INT32_FLOAT64 cimport BaseQRMUMPSSolver_INT32_FLOAT64

cimport numpy as cnp
cdef class NumpyQRMUMPSSolver_INT32_FLOAT64(BaseQRMUMPSSolver_INT32_FLOAT64):
    cpdef get_matrix_data(self, cnp.ndarray[cnp.int32_t, ndim=1] arow,
                                cnp.ndarray[cnp.int32_t, ndim=1] acol,
                                cnp.ndarray[cnp.float64_t, ndim=1] aval)
 
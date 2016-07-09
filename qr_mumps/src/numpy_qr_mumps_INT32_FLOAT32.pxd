from qr_mumps.src.qr_mumps_INT32_FLOAT32 cimport BaseQRMUMPSSolver_INT32_FLOAT32

cimport numpy as cnp
cdef class NumpyQRMUMPSSolver_INT32_FLOAT32(BaseQRMUMPSSolver_INT32_FLOAT32):
    cpdef get_matrix_data(self, cnp.ndarray[cnp.int32_t, ndim=1] arow,
                                cnp.ndarray[cnp.int32_t, ndim=1] acol,
                                cnp.ndarray[cnp.float32_t, ndim=1] aval)
 
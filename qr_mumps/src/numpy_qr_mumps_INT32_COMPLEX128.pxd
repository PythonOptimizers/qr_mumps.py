from qr_mumps.src.qr_mumps_INT32_COMPLEX128 cimport BaseQRMUMPSSolver_INT32_COMPLEX128

cimport numpy as cnp
cdef class NumpyQRMUMPSSolver_INT32_COMPLEX128(BaseQRMUMPSSolver_INT32_COMPLEX128):
    cpdef get_matrix_data(self, cnp.ndarray[cnp.int32_t, ndim=1] arow,
                                cnp.ndarray[cnp.int32_t, ndim=1] acol,
                                cnp.ndarray[cnp.complex128_t, ndim=1] aval)
 
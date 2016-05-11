from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX128_t cimport LLSparseMatrix_INT32_t_COMPLEX128_t
from cysparse.common_types.cysparse_types cimport *

from qr_mumps.src.qr_mumps_INT32_COMPLEX128 cimport BaseQRMUMPSSolver_INT32_COMPLEX128, c_to_fortran_index_array

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libc.stdint cimport int64_t
from libc.string cimport strncpy

cdef class CySparseQRMUMPSSolver_INT32_COMPLEX128(BaseQRMUMPSSolver_INT32_COMPLEX128):
    """
    QR_MUMPS Context.

    This version **only** deals with ``LLSparseMatrix_INT32_t_COMPLEX128_t`` objects.

    We follow the common use of QR_MUMPS. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in QR_MUMPS.

    Warning:
        The solver takes a "snapshot" of the matrix ``A``, i.e. the results given by the solver are only
        valid for the matrix given. If the matrix ``A`` changes aferwards, the results given by the solver won't
        reflect this change.

    """

    def __cinit__(self, int m, int n, int nnz, verbose=False):
        pass

    def __dealloc__(self):
        PyMem_Free(self.params.irn)
        PyMem_Free(self.params.jcn)
        PyMem_Free(self.params.val)

    cpdef get_matrix_data(self, LLSparseMatrix_INT32_t_COMPLEX128_t A):
        """
        Args:
            A: :class:`LLSparseMatrix_INT32_t_COMPLEX128_t` object.

        Note: we keep the same name for this method in all derived classes.
        """

        # CySparse matrices can have a `store_symmetric` flag set in that case
        # only lower triangular values are stored. However qr_mumps doesn't deal with it
        # we need to supply values for lower and upper parts.
        A.generalize()  # Convert matrix from symmetric to non-symmetric form (in-place).
        self.nnz = A.nnz
        self.params.nz = A.nnz

        print A

        # allocate memory for irn and jcn
        self.params.irn = <int *> PyMem_Malloc(self.nnz * sizeof(int))
        self.params.jcn = <int *> PyMem_Malloc(self.nnz * sizeof(int))

 
        a_val = <COMPLEX128_t *> PyMem_Malloc(self.nnz * sizeof(COMPLEX128_t))
        A.fill_triplet(self.params.irn, self.params.jcn, a_val)
        self.params.val = <double complex *> a_val

        # convert irn and jcn indices to Fortran format
        self.index_to_fortran()

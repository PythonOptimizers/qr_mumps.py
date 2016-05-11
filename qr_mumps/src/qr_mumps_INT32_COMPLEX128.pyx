"""
This is the interface to qr_mumps 

"""

from cpython.string cimport PyString_AsString

from libc.stdint cimport int64_t
from libc.string cimport strncpy, memcpy

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time

cdef extern from "complex.h":
    pass

cdef extern from "zqrm_mumps.h":
    cdef struct zqrm_spmat_type_c:
        int          *irn
        int          *jcn
        double complex *val
        int          m, n, nz
        int          *cperm_in
        int          icntl[20]
        double       rcntl[10]
        long int     gstats[10]
        int          h
    
    
    cdef double qrm_swtime();
    cdef void zqrm_spmat_init_c(zqrm_spmat_type_c *qrm_spmat_c);
    cdef void zqrm_spmat_destroy_c(zqrm_spmat_type_c *qrm_spmat_c);
    cdef void zqrm_readmat_c(char *matfile, zqrm_spmat_type_c *qrm_spmat_c);
    cdef void zqrm_analyse_c(zqrm_spmat_type_c *qrm_spmat_c, const char transp);
    cdef void zqrm_factorize_c(zqrm_spmat_type_c *qrm_spmat_c, const char transp);
    cdef void zqrm_solve_c(zqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          double complex *b, double complex *x, const int nrhs);
    cdef void zqrm_apply_c(zqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          double complex *b, const int nrhs);
    cdef void zqrm_matmul_c(zqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          const double complex alpha, double complex *x, 
                          const double complex beta, double complex *y, 
                          const int nrhs);
    cdef void zqrm_matnrm_c(zqrm_spmat_type_c *qrm_spmat_c, const char ntype, 
                          double *nrm);
    cdef void zqrm_vecnrm_c(const double complex *x, const int n, const int nrhs, 
                          const char ntype, double *nrm);
    cdef void zqrm_least_squares_c(zqrm_spmat_type_c *qrm_spmat_c, double complex *b, 
                          double complex *x, const int nrhs);
    cdef void zqrm_min_norm_c(zqrm_spmat_type_c *qrm_spmat_c, double complex *b, 
                                  double complex *x, const int nrhs);
    cdef void zqrm_residual_norm_c(zqrm_spmat_type_c *qrm_spmat_c, double complex *b, 
                                  double complex *x, const int nrhs, double *nrm);
    cdef void zqrm_residual_orth_c(zqrm_spmat_type_c *qrm_spmat_c, double complex *r, 
                                  const int nrhs, double *nrm);
    
    cdef void qrm_gseti_c(const char *string, int val);
    cdef void qrm_ggeti_c(const char *string, int *val);
    cdef void qrm_ggetii_c(const char *string, long long *val);
    
    cdef void zqrm_pseti_c(zqrm_spmat_type_c *qrm_spmat_c, const char *string, int val);
    cdef void zqrm_pgeti_c(zqrm_spmat_type_c *qrm_spmat_c, const char *string, int *val);
    cdef void zqrm_pgetii_c(zqrm_spmat_type_c *qrm_spmat_c, const char *string, long long *val);
    cdef void qrm_err_check_c();
    
    cdef enum icntl:
        qrm_ordering_
        qrm_sing_
        qrm_minamalg_
        qrm_nb_
        qrm_keeph_
        qrm_ib_
        qrm_rhsnb_
        qrm_rhsnthreads_
    
    cdef enum rcntl:
        qrm_amalgthr_
    
    cdef enum:
        qrm_auto=0
        qrm_natural_=1
        qrm_given_=2
        qrm_colamd_=3
        qrm_metis_=4
        qrm_scotch_=5
    
    cdef enum:
        qrm_e_facto_flops_=0
        qrm_e_nnz_r_=1
        qrm_e_nnz_h_=2
        qrm_facto_flops_=3
        qrm_nnz_r_=4
        qrm_nnz_h_=5
    
    cdef enum yn:
        qrm_no_=0
        qrm_yes_=1
    

cdef c_to_fortran_index_array(int * a, int a_size):
    cdef:
        int i

    for i from 0 <= i < a_size:
        a[i] += 1

cdef class AnalysisStatistics:
    """
    Statistics computed after an analysis phase has been performed.
    """
    def __cinit__(self, gstats, ordering, time):
        self.e_facto_flops = gstats[qrm_e_facto_flops_]
        self.e_nnz_r = gstats[qrm_e_nnz_r_]
        self.e_nnz_h = gstats[qrm_e_nnz_h_]
        self.ordering = ordering
        self.time = time

    def __str__(self):
        parts = ["Analysis statistics\n",
                 "-------------------\n",
                 "estimated number of nonzeros in matrix R: ",
                 str(self.e_nnz_r), "\n",
                 "estimated number of nonzeros in matrix H: ",
                 str(self.e_nnz_h), "\n",
                 "estimated number of flops: ", str(self.e_facto_flops), "\n",
                 "ordering used: ", ordering_name[self.ordering]]
        if hasattr(self, "time"):
            parts.extend(["\nanalysis time: ", str(self.time), " secs"])
        return "".join(parts)


cdef class FactorizationStatistics(object):
    """
    Statistics computed after a factorization phase has been performed.
    """
    def __cinit__(self, gstats, ordering, time):
        self.facto_flops = gstats[qrm_facto_flops_]
        self.nnz_r = gstats[qrm_nnz_r_]
        self.nnz_h = gstats[qrm_nnz_h_]
        self.ordering = ordering
        self.time = time

    def __str__(self):
        parts = ["Factorization statistics\n",
                 "------------------------\n",
                 "nonzeros in R matrix: ", str(self.nnz_r), "\n",
                 "nonzeros in H matrix: ", str(self.nnz_h), "\n",
                 "floating point operations: ", str(self.facto_flops), "\n", 
                 "ordering used: ", ordering_name[self.ordering]]
        if hasattr(self, "time"):
            parts.extend(["\nfactorization time: ", str(self.time), " secs"])
        return "".join(parts)


# QR_MUMPS possible ordering methods
orderings = { 'auto' : qrm_auto, 'natural' : qrm_natural_, 'given' : qrm_given_,
              'colamd' : qrm_colamd_, 'metis' : qrm_metis_, 'scotch' : qrm_scotch_, 'unset' : 6}

ordering_name = [ 'auto', 'natural', 'given',
                  'colamd', 'metis', 'scotch', 'unset']

# Base QR_MUMPS Solver
cdef class BaseQRMUMPSSolver_INT32_COMPLEX128:
    """
    Base QR_MUMPS Context.

    This version **only** deals with pointers.

    We follow the common use of QR_MUMPS. In particular, we use the same names for
    the methods of this class as their corresponding counter-parts in QR_MUMPS.
    """

    def __cinit__(self, int m, int n, int nnz, verbose=False):
        """
        Args:
            m: number of lines of matrix A
            n: number of columns of matrix A
            nnz: number of nonzeros of matrix A
            verbose: a boolean to turn on or off the verbosity of MUMPS
        """
        self.nrow = m
        self.ncol = n
        self.nnz = nnz

        # Initialize QR_MUMPS internal structure 
        zqrm_spmat_init_c(&self.params)

        self.params.m = self.nrow
        self.params.n = self.ncol
        self.params.nz = self.nnz
 
        if m < n:
            self.transp = 't' # True
        else:
            self.transp = 'n' # False

        self.analyzed = False
        self.factorized = False

        # set ordering to 'unset' : 6
        self.ordering = 6

        # set error level and output level 
        qrm_gseti_c("qrm_eunit", 6)
        qrm_gseti_c("qrm_ounit", 6)

        if not verbose:
            self.set_silent()


    cdef index_to_fortran(self):
        """
        Convert 0-based indices to Fortran indices (1-based).

        Note:
          Only for ``irn`` and ``jcn``.
        """

        # transform c index arrays to fortran arrays
        c_to_fortran_index_array(self.params.irn, self.nnz)
        c_to_fortran_index_array(self.params.jcn, self.nnz)


    def __dealloc__(self):
        # autodestruct qr_mumps internal
        zqrm_spmat_destroy_c(&self.params)

    # Properties
    property analyzed:
        def __get__(self): return self.analyzed
    property factorized:
        def __get__(self): return self.factorized
    property m:
        def __get__(self): return self.nrow
    property n:
        def __get__(self): return self.ncol
    property nnz:
        def __get__(self): return self.nnz
    property transp:
        def __get__(self): return self.transp
        def __set__(self, value): self.transp = value
    property gstats:
        def __get__(self): return self.params.gstats
    property analysis_statistics:
        def __get__(self):
            if not self.analyzed:
                raise Warning("``analyze`` hasn't been called yet!")
            return self.analysis_stats
    property factorization_statistics:
        def __get__(self):
            if not self.factorized:
                raise Warning("Factorization hasn't been done!")
            return self.factorization_stats

    def set_silent(self):
        """
        Silence **all** QR_MUMPS output.
        """
        qrm_gseti_c("qrm_ounit", 0)


    def analyze(self, ordering='auto'):
        """
        Performs analysis step of QR_MUMPS.

        TODO: ordering
        """
        if self.analyzed and self.ordering==orderings[ordering]:
            return

        # Set ordering option
        zqrm_pseti_c(&self.params, "qrm_ordering", orderings[ordering])

        t1 = time.clock()
        zqrm_analyse_c(&self.params, self.transp)
        t2 = time.clock()

        self.analyzed = True
        self.ordering = orderings[ordering]

        self.analysis_stats = AnalysisStatistics(self.params.gstats, self.ordering, t2 - t1)

    def factorize(self, ordering=None, pivot_tol=0.01):
        """
        Perform the QR factorization of the matrix A or A'.

        This factorization can then later be used to solve a linear system
        with `solve`. Statistical data of the factorization is stored in
        `factorization_stats`.

        Args:
            ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
                ordering to use in the factorization. The availability of a
                particular ordering depends on the MUMPS installation.  Default is
                'auto'.
            pivot_tol: number in the range [0, 1]
                pivoting threshold. Pivoting is typically limited in sparse
                solvers, as too much pivoting destroys sparsity. 1.0 means full
                pivoting, whereas 0.0 means no pivoting. Default is 0.01.
        """
        if ordering is not None:
            # Check if `self.ordering` was already initialized.
            # If it has been, factorized must be called with previous ordering, thus
            # a warning is issued to the user to tell him/her to rerun `analyze` with another
            # ordering option.
            if self.ordering<6 and orderings[ordering]!=self.ordering:
                raise Warning('Analyze was previously called with another ordering option. '+
                              'If you want to use another ordering option, you **need** to rerun `analyze`'+
                              'and supply the new ordering option.')
    
        # Analysis phase must be done before factorization
        if not self.analyzed:
            if ordering is None: ordering = 'auto'
            self.analyze(ordering=ordering)

        t1 = time.clock()
        zqrm_factorize_c(&self.params, self.transp)
        t2 = time.clock()

        self.factorized = True
        self.factorization_stats = FactorizationStatistics(self.params.gstats, self.ordering, t2 - t1)

    def solve(self, rhs, compute_residuals=False):
        """
        Args:
            rhs: dense NumPy array (matrix or vector).
            compute_residuals: boolean to compute the scaled norm of the residual (i.e. the normwise backward error).

        Returns:
            Dense NumPy array ``x`` (matrix or vector) with the solution(s) of the linear system
            or
            a tuple containing the solutions of the linear system and the scaled norm of the residual.
        """
        if rhs.dtype != np.complex128:
            raise TypeError("Type mismatch! Right hand side must be of type COMPLEX128")

        if not self.factorized:
            self.factorize()

        b = np.asfortranarray(rhs.copy())

        # test number of columns in rhs
        rhs_shape = rhs.shape
        if rhs.ndim==1:
            nrhs = 1 
            x = np.zeros(self.ncol, dtype=np.complex128) 
        elif rhs.ndim==2:
            nrhs = rhs_shape[1]
            x = np.zeros([self.ncol, nrhs], order='F', dtype=np.complex128)
        else:
            raise ValueError("Not implemented for 3 dimensional right-hand sides!")

        # Check on right-hand side number of lines
        if (rhs_shape[0] != self.nrow):
            raise ValueError("Right hand side has wrong size.\n"
                             "Attempting to solve the linear system, where A is of size (%d, %d) "
                             "and rhs is of size (%d,%d)"%(self.nrow, self.ncol, rhs_shape[0], nrhs))

        if self.transp=='n':
            zqrm_apply_c(&self.params, 't', <double complex *> cnp.PyArray_DATA(b), nrhs);
            zqrm_solve_c(&self.params, 'n', <double complex *> cnp.PyArray_DATA(b), <double complex *> cnp.PyArray_DATA(x), nrhs);
        elif self.transp=='t':
            zqrm_solve_c(&self.params, 't', <double complex *> cnp.PyArray_DATA(b), <double complex *> cnp.PyArray_DATA(x), nrhs);
            zqrm_apply_c(&self.params, 'n', <double complex *> cnp.PyArray_DATA(x), nrhs);


        # compute residuals if requested
        if compute_residuals:
            residuals = np.asfortranarray(rhs.copy())
            residual_norm = np.zeros(nrhs, dtype=np.float64)
            zqrm_residual_norm_c(&self.params,
                                                             <double complex *> cnp.PyArray_DATA(residuals),
                                                             <double complex *> cnp.PyArray_DATA(x),
                                                             nrhs,
                                                             <double *>cnp.PyArray_DATA(residual_norm))
            return (x, residuals, residual_norm)
        else:
            return x


    def least_squares(self, rhs):
        if self.m < self.n:
            raise RuntimeError("least_squares method can only be called for matrix A with more lines than columns")

        b = np.asfortranarray(rhs.copy())

        # test number of columns in rhs
        rhs_shape = rhs.shape
        if rhs.ndim==1:
            nrhs = 1 
            x = np.zeros(self.ncol, dtype=np.complex128) 
        elif rhs.ndim==2:
            nrhs = rhs_shape[1]
            x = np.zeros([self.ncol, nrhs], order='F', dtype=np.complex128)
        else:
            raise ValueError("Not implemented for 3 dimensional right-hand sides!")

        # Check on right-hand side number of lines
        if (rhs_shape[0] != self.nrow):
            raise ValueError("Right hand side has wrong size.\n"
                             "Attempting to solve the linear system, where A is of size (%d, %d) "
                             "and rhs is of size (%d,%d)"%(self.nrow, self.ncol, rhs_shape[0], nrhs))

        zqrm_least_squares_c(&self.params, <double complex *> cnp.PyArray_DATA(b), <double complex *> cnp.PyArray_DATA(x), nrhs)
        return x

    def minimum_norm(self, rhs):
        if self.m >= self.n:
            raise RuntimeError("minimum_norm method can only be called for matrix A with more columns than lines")

        b = np.asfortranarray(rhs.copy())

        # test number of columns in rhs
        rhs_shape = rhs.shape
        if rhs.ndim==1:
            nrhs = 1 
            x = np.zeros(self.ncol, dtype=np.complex128) 
        elif rhs.ndim==2:
            nrhs = rhs_shape[1]
            x = np.zeros([self.ncol, nrhs], order='F', dtype=np.complex128)
        else:
            raise ValueError("Not implemented for 3 dimensional right-hand sides!")

        # Check on right-hand side number of lines
        if (rhs_shape[0] != self.nrow):
            raise ValueError("Right hand side has wrong size.\n"
                             "Attempting to solve the linear system, where A is of size (%d, %d) "
                             "and rhs is of size (%d,%d)"%(self.nrow, self.ncol, rhs_shape[0], nrhs))

        zqrm_min_norm_c(&self.params, <double complex *> cnp.PyArray_DATA(b), <double complex *> cnp.PyArray_DATA(x), nrhs)
        return x


    cpdef cnp.ndarray[cnp.complex128_t] refine(self, cnp.ndarray[cnp.complex128_t] x, cnp.ndarray[cnp.complex128_t] rhs, int niter):
        """
        Let x be the initial solution of Ax = b
        Compute residual r = b - Ax
        for i = 1 to t do
            Solve A Dx = r using the computed factorization
            x = x + Dx
            r = b - Ax
        end for
        """
        
        cdef int i

#TODO: test if factorise has been called

        residuals = np.asfortranarray(rhs.copy())
        new_x =  np.asfortranarray(x.copy())

        # test number of columns in rhs
        rhs_shape = rhs.shape
        if rhs.ndim==1:
            nrhs = 1 
        elif rhs.ndim==2:
            nrhs = rhs_shape[1]
        else:
            raise ValueError("Not implemented for 3 dimensional right-hand sides!")
                        
        x_shape = x.shape
        if x.ndim==1:
            nx = 1 
        elif x.ndim==2:
            nx = x_shape[1]
 
        assert(nx, nrhs)

        for i in xrange(0,niter):
            zqrm_matmul_c(&self.params, 'n', <double> -1,
                                                      <double complex *> cnp.PyArray_DATA(new_x), 
                                                      <double> 1,
                                                      <double complex *> cnp.PyArray_DATA(residuals),
                                                      nrhs)

            if self.transp=='n':
                zqrm_apply_c(&self.params, 't', <double complex *> cnp.PyArray_DATA(residuals), nrhs);
                zqrm_solve_c(&self.params, 'n', <double complex *> cnp.PyArray_DATA(residuals), <double complex *> cnp.PyArray_DATA(new_x), nrhs);
            elif self.transp=='t':
                zqrm_solve_c(&self.params, 't', <double complex *> cnp.PyArray_DATA(residuals), <double complex *> cnp.PyArray_DATA(new_x), nrhs);
                zqrm_apply_c(&self.params, 'n', <double complex *> cnp.PyArray_DATA(new_x), nrhs);

            new_x = x + new_x
            x = new_x.copy()
       
        return new_x


cimport numpy as cnp

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
        int          h; 
    
    
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
    
    cdef enum gstats:
        qrm_e_facto_flops_=0
        qrm_e_nnz_r_=1
        qrm_e_nnz_h_=2
        qrm_facto_flops_=3
        qrm_nnz_r_=4
        qrm_nnz_h_=5
    
    cdef enum yn:
        qrm_no_=0
        qrm_yes_=1
    

cdef class AnalysisStatistics:
    cdef :
        long int e_facto_flops
        long int e_nnz_r
        long int e_nnz_h
        int ordering
        double time


cdef class FactorizationStatistics:
    cdef :
        long int facto_flops
        long int nnz_r
        long int nnz_h
        int ordering
        double time


cdef c_to_fortran_index_array(int * a, int a_size)

cdef class BaseQRMUMPSSolver_INT32_COMPLEX128:
    cdef:

        int nrow
        int ncol
        int nnz

        # QR_MUMPS internal structure
        zqrm_spmat_type_c params
 
        char transp 

        bint analyzed
        bint factorized
        bint out_of_core

        int ordering

        AnalysisStatistics analysis_stats
        FactorizationStatistics factorization_stats
        object solve_stats

    cdef index_to_fortran(self)

    cpdef cnp.ndarray[cnp.complex128_t] refine(self, cnp.ndarray[cnp.complex128_t] x, cnp.ndarray[cnp.complex128_t] rhs, int niter)
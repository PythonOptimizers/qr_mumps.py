cimport numpy as cnp

cdef extern from "sqrm_mumps.h":
    cdef struct sqrm_spmat_type_c:
        int          *irn
        int          *jcn
        float *val
        int          m, n, nz
        int          *cperm_in
        int          icntl[20]
        double       rcntl[10]
        long int     gstats[10]
        int          h; 
    
    
    cdef double qrm_swtime();
    cdef void sqrm_spmat_init_c(sqrm_spmat_type_c *qrm_spmat_c);
    cdef void sqrm_spmat_destroy_c(sqrm_spmat_type_c *qrm_spmat_c);
    cdef void sqrm_readmat_c(char *matfile, sqrm_spmat_type_c *qrm_spmat_c);
    cdef void sqrm_analyse_c(sqrm_spmat_type_c *qrm_spmat_c, const char transp);
    cdef void sqrm_factorize_c(sqrm_spmat_type_c *qrm_spmat_c, const char transp);
    cdef void sqrm_solve_c(sqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          float *b, float *x, const int nrhs);
    cdef void sqrm_apply_c(sqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          float *b, const int nrhs);
    cdef void sqrm_matmul_c(sqrm_spmat_type_c *qrm_spmat_c, const char transp,
                          const float alpha, float *x, 
                          const float beta, float *y, 
                          const int nrhs);
    cdef void sqrm_matnrm_c(sqrm_spmat_type_c *qrm_spmat_c, const char ntype, 
                          float *nrm);
    cdef void sqrm_vecnrm_c(const float *x, const int n, const int nrhs, 
                          const char ntype, float *nrm);
    cdef void sqrm_least_squares_c(sqrm_spmat_type_c *qrm_spmat_c, float *b, 
                          float *x, const int nrhs);
    cdef void sqrm_min_norm_c(sqrm_spmat_type_c *qrm_spmat_c, float *b, 
                                  float *x, const int nrhs);
    cdef void sqrm_residual_norm_c(sqrm_spmat_type_c *qrm_spmat_c, float *b, 
                                  float *x, const int nrhs, float *nrm);
    cdef void sqrm_residual_orth_c(sqrm_spmat_type_c *qrm_spmat_c, float *r, 
                                  const int nrhs, float *nrm);
    
    cdef void qrm_gseti_c(const char *string, int val);
    cdef void qrm_ggeti_c(const char *string, int *val);
    cdef void qrm_ggetii_c(const char *string, long long *val);
    
    cdef void sqrm_pseti_c(sqrm_spmat_type_c *qrm_spmat_c, const char *string, int val);
    cdef void sqrm_pgeti_c(sqrm_spmat_type_c *qrm_spmat_c, const char *string, int *val);
    cdef void sqrm_pgetii_c(sqrm_spmat_type_c *qrm_spmat_c, const char *string, long long *val);
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

cdef class BaseQRMUMPSSolver_INT32_FLOAT32:
    cdef:

        int nrow
        int ncol
        int nnz

        # QR_MUMPS internal structure
        sqrm_spmat_type_c params
 
        char transp 

        bint analyzed
        bint factorized
        bint out_of_core

        int ordering

        AnalysisStatistics analysis_stats
        FactorizationStatistics factorization_stats
        object solve_stats

    cdef index_to_fortran(self)

    cpdef cnp.ndarray[cnp.float32_t] refine(self, cnp.ndarray[cnp.float32_t] x, cnp.ndarray[cnp.float32_t] rhs, int niter)
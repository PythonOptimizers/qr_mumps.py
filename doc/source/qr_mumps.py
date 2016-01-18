"""
This is the interface to qr_mumps

"""


class AnalysisStatistics:
    """
    Statistics computed after an analysis phase has been performed.
    """
    def __init__(self, params, ordering, time):
        self.e_facto_flops = params[qrm_e_facto_flops_]
        self.e_nnz_r = params[qrm_e_nnz_r_]
        self.e_nnz_h = params[qrm_e_nnz_h_]
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


class FactorizationStatistics(object):
    """
    Statistics computed after a factorization phase has been performed.
    """
    def __init__(self, params, ordering, time):
        self.facto_flops = params[qrm_facto_flops_]
        self.nnz_r = params[qrm_nnz_r_]
        self.nnz_h = params[qrm_nnz_h_]
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
class BaseQRMUMPSSolver_@index@_@type@:
    """
    Base QR_MUMPS Context.

    This version **only** deals with pointers.

    We follow the common use of QR_MUMPS. In particular, we use the same names for
    the methods of this class as their corresponding counter-parts in QR_MUMPS.
    """

    def __init__(self, @index|generic_to_c_type@ m, @index|generic_to_c_type@ n, @index|generic_to_c_type@ nnz, verbose=False):
        """
        Args:
            m: number of lines of matrix A
            n: number of columns of matrix A
            nnz: number of nonzeros of matrix A
            verbose: a boolean to turn on or off the verbosity of MUMPS
        """
        pass

    def index_to_fortran(self):
        """
        Convert 0-based indices to Fortran indices (1-based).

        Note:
          Only for ``irn`` and ``jcn``.
        """
        pass

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
        pass


    def analyze(self, ordering='auto'):
        """
        Performs analysis step of QR_MUMPS.

        TODO: ordering
        """
        pass

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
        pass

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
        pass

    def least_squares(self, rhs):
        pass

    def minimum_norm(self, rhs):
        pass

    cpdef cnp.ndarray[cnp.@type|lower@_t] refine(self, cnp.ndarray[cnp.@type|lower@_t] x, cnp.ndarray[cnp.@type|lower@_t] rhs, @index|generic_to_c_type@ niter):
        """
        Let x be the initial solution of Ax = b
        Compute residual r = b - Ax
        for i = 1 to t do
            Solve A Dx = r using the computed factorization
            x = x + Dx
            r = b - Ax
        end for
        """
        pass


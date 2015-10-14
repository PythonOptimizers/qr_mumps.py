"""
Factory method to access qr_mumps.
"""
import numpy as np

{% for index_type in qr_mumps_index_list %}
    {% for element_type in qr_mumps_type_list %}
from qr_mumps.src.qr_mumps_@index_type@_@element_type@ import Numpyqr_mumpsContext_@index_type@_@element_type@
    {% endfor %}
{% endfor %}

cysparse_installed = False
try:
{% for index_type in qr_mumps_index_list %}
    {% for element_type in qr_mumps_type_list %}
    from qr_mumps.src.cysparse_qr_mumps_@index_type@_@element_type@ import CySparseqr_mumpsContext_@index_type@_@element_type@
    {% endfor %}
    from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
    from cysparse.types.cysparse_types import *
    cysparse_installed = True
{% endfor %}
except:
    pass

allowed_types = '\titype:
{%- for index_name in qr_mumps_index_list -%}
    @index_name@
    {%- if index_name != qr_mumps_index_list|last -%}
    ,
    {%- endif -%}
{%- endfor -%}
\n\tdtype:
{%- for element_name in qr_mumps_type_list -%}
    @element_name@
    {%- if element_name != qr_mumps_type_list|last -%}
    ,
    {%- endif -%}
{%- endfor -%}
\n'
type_error_msg = 'Matrix has an index and/or element type that is incompatible with qr_mumps\nAllowed types:\n%s' % allowed_types

def qr_mumpsContext(arg1, verbose=False):
    """
    Create and return the right qr_mumps context based on the element type
    supplied as input.

    qr_mumps ("MUltifrontal Massively Parallel Solver") is a package for solving systems
    of linear equations of the form Ax = b, where A is a square **sparse** matrix that can be
    either unsymmetric, symmetric positive definite, or general symmetric, on distributed
    memory computers. 
    
    qr_mumps performs a Gaussian factorization
      A = LU
    where L is a lower triangular matrix and U an upper triangular matrix.

    If the matrix is symmetric then qr_mumps performs the factorization
      A = LDL^T 
    where D is block diagonal matrix.
    
    Args:
        n: size of matrix A
        a_row: row indices of non zero elements of A
        a_col: column indices of non zero elements of A
        a_val: values of non zeros elements of A
        sym:   a boolean indicating if A is a symmetric matrix or not
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

{% for index_type in qr_mumps_index_list %}
  {% if index_type == qr_mumps_index_list |first %}
        if itype == np.@index_type|lower@:
      {% for element_type in qr_mumps_type_list %}
        {% if element_type == qr_mumps_type_list |first %}
            if dtype == np.@element_type|lower@:
        {% else %}
            elif dtype == np.@element_type|lower@:
        {% endif %}
                return Numpyqr_mumpsContext_@index_type@_@element_type@(m, n, a_row, a_col, a_val, verbose=verbose)
      {% endfor %}
  {% else %}
        elif itype == np.@index_type|lower@:
      {% for element_type in qr_mumps_type_list %}
        {% if element_type == qr_mumps_type_list |first %}
            if dtype == np.@element_type|lower@:
        {% else %}
            elif dtype == np.@element_type|lower@:
        {% endif %}
                return Numpyqr_mumpsContext_@index_type@_@element_type@(m, n, a_row, a_col, a_val, verbose=verbose)
      {% endfor %}
  {% endif %}
{% endfor %}
        else:
            raise TypeError(type_error_msg)

    elif cysparse_installed:
        if not PyLLSparseMatrix_Check(arg1):
            raise TypeError('arg1 should be a LLSparseMatrix')

        A = arg1
        itype = A.itype
        dtype = A.dtype

{% for index_type in qr_mumps_index_list %}
    {% if index_type == qr_mumps_index_list |first %}
        if itype == @index_type@_T:
    {% for element_type in qr_mumps_type_list %}
        {% if element_type == qr_mumps_type_list |first %}
            if dtype == @element_type@_T:
        {% else %}
            elif dtype == @element_type@_T:
        {% endif %}
                return CySparseqr_mumpsContext_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% else %}
        elif itype == @index_type@_T:
    {% for element_type in qr_mumps_type_list %}
        {% if element_type == qr_mumps_type_list |first %}
            if dtype == @element_type@_T:
        {% else %}
            elif dtype == @element_type@_T:
        {% endif %}
                return CySparseqr_mumpsContext_@index_type@_@element_type@(A, verbose=verbose)
    {% endfor %}
    {% endif %}
{% endfor %}
        else:
            raise TypeError(type_error_msg)


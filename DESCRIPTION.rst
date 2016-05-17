qr_mumps.py
===========

Cython/Python inferface to qr_mumps (`A multithreaded multifrontal QR solver <http://buttari.perso.enseeiht.fr/qr_mumps/>`_).

It supports all four types supported by qr_mumps (single real, double real, single complex and double complex).

Dependencies
------------

For the common user (Python version):

- `Numpy <http://www.numpy.org>`_

For the advanced user (Cython version), include everything needed for the Python version and add:

- `Cython <https://github.com/cython/cython.git>`_
- `cygenja <https://github.com/PythonOptimizers/cygenja.git>`_

If you intend to generate the documention:

- Sphinx
- sphinx_bootstrap_theme.

To run the tests:

- nose.

Optional dependencies
---------------------

`qr_mumps.py` provides facilities for sparse matrices coming from the `CySparse <https://github.com/PythonOptimizers/cysparse>`_ library.
If you want to use these facilities, set the location of the `CySparse` library in your `site.cfg` file.

..  qr_mumps_intallation:

===================================
Installation
===================================

:program:`qr_mumps.py` uses external packages that are not included in the :program:`qr_mumps.py`
source code because they are released under different licenses than the one used for
:program:`qr_mumps.py`. Hence we cannot distribute their code and you have to obtain them yourself.

:program:`qr_mumps` source code
===============================
You need to install :program:`qr_mumps`.
Follow instructions on `their website <http://buttari.perso.enseeiht.fr/qr_mumps/>`_.

If you are under **OS X**, a `Homebrew <http://brew.sh>`_ formula is available.
Follow the instructions to install :program:`Homebrew`.
Then, :program:`qr_mumps` and its dependencies can be installed automatically in `/usr/local` by typing

  .. code-block:: bash

     brew tap homebrew/science

     brew install qr_mumps


:program:`qr_mumps` supports several permutation methods (ordering methods) through third party packages:
:program:`COLAMD` (not supported in the :program:`Homebrew` formula), :program:`SCOTCH` (add the `with-scotch5`) and :program:`Metis` (default on :program:`Homebrew`).


:program:`Python` interfaces
============================

:program:`qr_mumps.py` installation is done in few simple steps:

1. Clone the repository:

  ..  code-block:: bash

      git clone https://github.com/PythonOptimizers/qr_mumps.py.git


2. Install the :program:`Python` dependency:

- :program:`NumPy`

  Python installer :program:`pip` is recommended for that

  ..  code-block:: bash

      pip install numpy

3. Copy :file:`site.template.cfg` to :file:`site.cfg` and adjust it to reflect your own environment

4. Compile and install the library:

  The preferred way to install the library is to install it in its own `virtualenv`.

  To compile and install the library, just type

      ..  code-block:: bash

          python setup.py install

Optional dependency
===================

:program:`qr_mumps.py` provides facilities for sparse matrices coming from the `CySparse <https://github.com/PythonOptimizers/cysparse>`_ library.
If you want to use these facilities, set the location of the :program:`CySparse` library in your `site.cfg` file.


Further dependencies
====================

Documentation
-------------

To generate the documentation you will need other Python dependencies:

- :program:`Sphinx`
- :program:`sphinx-bootstrap-theme`

which can be easily installed using :program:`pip`


Testing
-------
Testing is done using :program:`nose`, so it needs to be installed before running them.


Note that a complete list of dependencies is provided in the :file:`requirements.txt` file. You can easily install all of them with:

..  code-block:: bash

    pip install -r requirements.txt


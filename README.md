# qr_mumps.py

[![Build Status](https://travis-ci.com/PythonOptimizers/qr_mumps.py.svg?token=33z5zptBt5SzXC4ZvLpF&branch=without-cython)](https://travis-ci.com/PythonOptimizers/qr_mumps.py)

Cython/Python inferface to qr_mumps ([A multithreaded multifrontal QR solver](http://buttari.perso.enseeiht.fr/qr_mumps/)).

It supports all four types (single real, double real, single complex and double complex).

## Dependencies

For the Python version:

- [`Numpy`](http://www.numpy.org) 

For the Cython version, include everything needed for the Python version and add:

- [`Cython`](https://github.com/cython/cython.git)
- [`cygenja`](https://github.com/PythonOptimizers/cygenja.git)

If you intend to generate the documention:

- Sphinx
- sphinx_bootstrap_theme.

To run the tests:

- nose.

## Optional dependencies

`qr_mumps.py` provides facilities for sparse matrices coming from the [`CySparse`](https://github.com/PythonOptimizers/cysparse) library.
If you want to use these facilities, set the location of the `CySparse` library in your `site.cfg` file.


## Installation

1. You need to install qr_mumps. Follow instructions on [their website](http://buttari.perso.enseeiht.fr/qr_mumps/).
       If you are under OS X, a [Homebrew](http://brew.sh) formula is available. Follow the instructions to install Homebrew.
       Then, qr_mumps and its dependencies can be installed automatically in `/usr/local` by typing

    	brew tap homebrew/science

    	brew install qr_mumps

2. Clone this repo

3. Copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration
    
3. Install `qr_mumps.py`

   - Python version

	        python setup.py install

   - Cython version

            python generate_code.py
    	    python setup.py build
    	    python setup.py install

## Running examples

## TODO:

  - [ ] Add refine method
  - [ ] Add a Q-less derived class
  - [ ] Update examples
  - [ ] Update docstrings
  - [ ] ensure all code is PEP8 and PEP257 compliant

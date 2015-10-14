# qr_mumps.py
Python inferface to qr_mumps ([A multithreaded multifrontal QR solver](http://buttari.perso.enseeiht.fr/qr_mumps/)).

It supports all four types (single real, double real, single complex and double complex).


## Installation
    
1. You need to install qr_mumps. Follow instructions on [their website](http://buttari.perso.enseeiht.fr/qr_mumps/).
       If you are under OS X, a [Homebrew](http://brew.sh) formula is available. Follow the instructions to install Homebrew.
       Then, MUMPS and its dependencies can be installed automatically in `/usr/local` by typing

    	brew install gcc  # contains gfortran

    	brew tap homebrew/science

    	brew install qr_mumps

2. Clone this repo and modify the `site.cfg` to match your configuration
    
3. Install `qr_mumps.py`

    	python generate_code.py -a
    	python setup.py build
    	python setup.py install

## Running examples

## TODO:

  - [ ] Everything 
  - [ ] Add tests for everything
  - [ ] ensure all code is PEP8 and PEP257 compliant

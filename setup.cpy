#!/usr/bin/env python

# The file setup.py is automatically generated
# Generate it with
# python generate_code -s

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

import ConfigParser
import os
import copy

from codecs import open
from os import path

########################################################################################################################
# INIT
########################################################################################################################
qr_mumps_config = ConfigParser.SafeConfigParser()
qr_mumps_config.read('site.cfg')

version = {}
with open("qr_mumps/version.py") as fp:
      exec(fp.read(), version)
# later on we use: version['version']

numpy_include = np.get_include()

# DEFAULT
default_include_dir = qr_mumps_config.get('DEFAULT', 'include_dirs').split(os.pathsep)
default_library_dir = qr_mumps_config.get('DEFAULT', 'library_dirs').split(os.pathsep)

# qr_mumps
qr_mumps_compiled_in_64bits = qr_mumps_config.getboolean('QR_MUMPS', 'qr_mumps_compiled_in_64bits')

# find user defined directories
qr_mumps_include_dirs = qr_mumps_config.get('QR_MUMPS', 'include_dirs').split(os.pathsep)
if qr_mumps_include_dirs == '':
    qr_mumps_include_dirs = default_include_dir
qr_mumps_library_dirs = qr_mumps_config.get('QR_MUMPS', 'library_dirs').split(os.pathsep)
if qr_mumps_library_dirs == '':
    qr_mumps_library_dirs = default_library_dir
           
# OPTIONAL
build_cysparse_ext = False           
if qr_mumps_config.has_section('CYSPARSE'):
    build_cysparse_ext = True
    cysparse_rootdir = qr_mumps_config.get('CYSPARSE', 'cysparse_rootdir').split(os.pathsep)
    if cysparse_rootdir == '':
        raise ValueError("You must specify where CySparse source code is" +
                         "located. Use `cysparse_rootdir` to specify its path.")


########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...
ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
ext_params['extra_link_args'] = []


context_ext_params = copy.deepcopy(ext_params)
qr_mumps_ext = []
{% for index_type in index_list %}
  {% for element_type in type_list %}
base_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
base_ext_params_@index_type@_@element_type@['include_dirs'].extend(qr_mumps_include_dirs)
base_ext_params_@index_type@_@element_type@['library_dirs'] = qr_mumps_library_dirs
base_ext_params_@index_type@_@element_type@['libraries'] = [] # 'scalapack', 'pord']
base_ext_params_@index_type@_@element_type@['libraries'].append('@element_type|numpy_to_qr_mumps_type@qrm')
base_ext_params_@index_type@_@element_type@['libraries'].append('qrm_common')
qr_mumps_ext.append(Extension(name="qr_mumps.src.qr_mumps_@index_type@_@element_type@",
                sources=['qr_mumps/src/qr_mumps_@index_type@_@element_type@.pxd',
                'qr_mumps/src/qr_mumps_@index_type@_@element_type@.pyx'],
                **base_ext_params_@index_type@_@element_type@))

numpy_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
numpy_ext_params_@index_type@_@element_type@['include_dirs'].extend(qr_mumps_include_dirs)
qr_mumps_ext.append(Extension(name="qr_mumps.src.numpy_qr_mumps_@index_type@_@element_type@",
                 sources=['qr_mumps/src/numpy_qr_mumps_@index_type@_@element_type@.pxd',
                 'qr_mumps/src/numpy_qr_mumps_@index_type@_@element_type@.pyx'],
                 **numpy_ext_params_@index_type@_@element_type@))

  {% endfor %}
{% endfor %}

if build_cysparse_ext:
{% for index_type in index_list %}
  {% for element_type in type_list %}
    cysparse_ext_params_@index_type@_@element_type@ = copy.deepcopy(ext_params)
    cysparse_ext_params_@index_type@_@element_type@['include_dirs'].extend(cysparse_rootdir)
    cysparse_ext_params_@index_type@_@element_type@['include_dirs'].extend(qr_mumps_include_dirs)
    qr_mumps_ext.append(Extension(name="qr_mumps.src.cysparse_qr_mumps_@index_type@_@element_type@",
                 sources=['qr_mumps/src/cysparse_qr_mumps_@index_type@_@element_type@.pxd',
                 'qr_mumps/src/cysparse_qr_mumps_@index_type@_@element_type@.pyx'],
                 **cysparse_ext_params_@index_type@_@element_type@))

  {% endfor %}
{% endfor %}


packages_list = ['qr_mumps', 'qr_mumps.src', 'tests']


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name=  'qr_mumps.py',
      version=version['version'],
      description='A python interface to qr_mumps',
      long_description=long_description,
      #Author details
      author='Nikolaj van Omme, Sylvain Arreckx, Dominique Orban',
{% raw %}
      author_email='qr_mumps\@TODO.com',
{% endraw %}
      maintainer = "qr_mumps.py Developers",
{% raw %}
      maintainer_email = "dominique.orban@gerad.ca",
{% endraw %}
      summary = "A python interface to qr_mumps",
      url = "https://github.com/optimizers/qr_mumps",
      download_url = "https://github.com/optimizers/qr_mumps",
      license='LGPL',
      classifiers=filter(None, CLASSIFIERS.split('\n')),
      install_requires=['numpy', 'Cython'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = qr_mumps_ext,
      package_dir = {"qr_mumps": "qr_mumps"},
      packages=packages_list,
      zip_safe=False
)


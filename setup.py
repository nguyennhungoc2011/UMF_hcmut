#! /usr/bin/env python
'''
  Copyright (C) 2016, WS-DREAM, CUHK
  License: MIT
  Updated: 2024/5/30 by Ngoc Nhu Nguyen
'''

description = 'UMF - A Python Package to Updated Adaptive Matrix Factorization'

from distutils.core import setup, Extension
import os
import os.path
import numpy
from distutils.sysconfig import *

try:
   from distutils.command.build_py import build_py_2to3 \
       as build_py
except ImportError:
   from distutils.command.build_py import build_py

try:
   from Cython.Distutils import build_ext
except ImportError:
   use_cython = False
else:
   use_cython = True

extra_compile_args = ['-O2']

#### data files
data_files = []

#### scripts
scripts = []

#### Python include
py_inc = [get_python_inc()]

#### NumPy include
np_inc = [numpy.get_include()]

#### cmdclass
cmdclass = {'build_py': build_py}

#### Extension modules
ext_modules = []
if use_cython:
    cmdclass.update({'build_ext': build_ext})
    ext_modules += [Extension("UMF.UMF", 
                              ["UMF/c_UMF.cpp",
                              "UMF/UMF.pyx"],
                              language='c++',
                              include_dirs=py_inc + np_inc)
                              ]

else:
    ext_modules += [Extension("UMF.UMF", 
                              ["UMF/c_UMF.cpp",
                              "UMF/UMF.cpp"],
                              include_dirs=py_inc + np_inc)
                              ]

packages=['UMF']

classifiers = ['Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT',
               'Programming Language :: C++',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering :: Artificial Intelligence'
               ]

setup(name = 'UMF',
      version='1.0',
      requires=['numpy (>=1.8.1)', 'scipy (>=0.13.3)'],
      description=description,
      packages=packages,
      license='MIT',
      classifiers=classifiers,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      scripts=scripts,
      data_files=data_files
      )

print('==============================================')
print('Setup succeeded!\n')


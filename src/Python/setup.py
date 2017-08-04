#!/usr/bin/env python

import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import sys
import numpy
import platform	

cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)

library_include_path = ""
library_lib_path = ""
try:
    library_include_path = os.environ['LIBRARY_INC']
    library_lib_path = os.environ['LIBRARY_LIB']
except:
    library_include_path = os.environ['PREFIX']+'/include'
    pass
    
extra_include_dirs = [numpy.get_include(), library_include_path]
extra_library_dirs = [library_include_path+"/../lib", "C:\\Apps\\Miniconda2\\envs\\cil27\\Library\\lib"]
extra_compile_args = ['-fopenmp','-O2', '-funsigned-char', '-Wall', '-std=c++0x']
extra_libraries = []
if platform.system() == 'Windows':
    extra_compile_args[0:] = ['/DWIN32','/EHsc','/DBOOST_ALL_NO_LIB' , '/openmp' ]   
    extra_include_dirs += ["..\\..\\main_func\\regularizers_CPU\\","."]
    if sys.version_info.major == 3 :   
        extra_libraries += ['boost_python3-vc140-mt-1_64', 'boost_numpy3-vc140-mt-1_64']
    else:
        extra_libraries += ['boost_python-vc90-mt-1_64', 'boost_numpy-vc90-mt-1_64']
else:
    extra_include_dirs += ["../../main_func/regularizers_CPU","."]
    if sys.version_info.major == 3:
        extra_libraries += ['boost_python3', 'boost_numpy3','gomp']
    else:
        extra_libraries += ['boost_python', 'boost_numpy','gomp']

setup(
    name='ccpi',
	description='CCPi Core Imaging Library - FISTA Reconstruction Module',
	version=cil_version,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("regularizers",
                             sources=["fista_module.cpp",
                                      "..\\..\\main_func\\regularizers_CPU\\FGP_TV_core.c",
                                      "..\\..\\main_func\\regularizers_CPU\\SplitBregman_TV_core.c",
                                      "..\\..\\main_func\\regularizers_CPU\\LLT_model_core.c",
                                      "..\\..\\main_func\\regularizers_CPU\\utils.c"
                                        ],
                             include_dirs=extra_include_dirs, library_dirs=extra_library_dirs, extra_compile_args=extra_compile_args, libraries=extra_libraries ), 
    
    ],
	zip_safe = False,	
	packages = {'ccpi','ccpi.reconstruction'},
)

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
#extra_library_dirs = [os.path.join(library_include_path, "..", "lib")]
extra_compile_args = []
extra_library_dirs = []
extra_compile_args = []
extra_link_args = []
extra_libraries = ['cilreg']

extra_include_dirs += [os.path.join(".." , ".." , "Core",  "regularizers_CPU"),
                       	   os.path.join(".." , ".." , "Core",  "regularizers_GPU") , 
						   "."]

if platform.system() == 'Windows':
    
						   
    extra_compile_args[0:] = ['/DWIN32','/EHsc','/DBOOST_ALL_NO_LIB' , '/openmp' ]   
    
    if sys.version_info.major == 3 :   
        extra_libraries += ['boost_python3-vc140-mt-1_64', 'boost_numpy3-vc140-mt-1_64']
    else:
        extra_libraries += ['boost_python-vc90-mt-1_64', 'boost_numpy-vc90-mt-1_64']
else:
    extra_compile_args = ['-fopenmp','-O2', '-funsigned-char', '-Wall', '-std=c++0x']
    if sys.version_info.major == 3:
        extra_libraries += ['boost_python3', 'boost_numpy3','gomp']
    else:
        extra_libraries += ['boost_python', 'boost_numpy','gomp']

setup(
    name='ccpi',
	description='CCPi Core Imaging Library - Image Regularizers',
	version=cil_version,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ccpi.imaging.cpu_regularizers",
                             sources=[os.path.join("." , "fista_module.cpp" ),
                                     # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "FGP_TV_core.c"),
									 # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "SplitBregman_TV_core.c"),
									 # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "LLT_model_core.c"),
									 # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "PatchBased_Regul_core.c"),
									 # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "TGV_PD_core.c"),
									 # os.path.join("@CMAKE_SOURCE_DIR@" , "main_func" ,  "regularizers_CPU", "utils.c")
                                        ],
                             include_dirs=extra_include_dirs, 
							 library_dirs=extra_library_dirs, 
							 extra_compile_args=extra_compile_args, 
							 libraries=extra_libraries ), 
    
    ],
	zip_safe = False,	
	packages = {'ccpi','ccpi.imaging'},
)



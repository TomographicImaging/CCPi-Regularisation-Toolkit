#   Copyright 2018 Edoardo Pasca
cmake_minimum_required (VERSION 3.0)

project(regulariserPython)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.

set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)

# conda orchestrated build
message("CIL_VERSION ${CIL_VERSION}")
#include (GenerateExportHeader)

find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
endif()

	
## Build the regularisers package as a library
message("Creating Regularisers as shared library")

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /openmp /DCCPiCore_EXPORTS")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES)
		
  message("library lib: ${LIBRARY_LIB}")
  
elseif(UNIX)
   set (FLAGS "-fopenmp -O2 -funsigned-char -Wall  -Wl,--no-undefined  -DCCPiReconstructionIterative_EXPORTS -std=c++0x")  
   set (EXTRA_LIBRARIES 
		"gomp"
		)
endif()

# GPU regularisers

find_package(CUDA)
if (CUDA_FOUND)
  message("CUDA FOUND")
  set (SETUP_GPU_WRAPPERS "extra_libraries += ['cilregcuda']\n\
setup( \n\
    name='ccpi', \n\
 	description='CCPi Core Imaging Library - Image regularisers GPU',\n\
 	version=cil_version,\n\
    cmdclass = {'build_ext': build_ext},\n\
    ext_modules = [Extension('ccpi.filters.gpu_regularisers',\n\
                              sources=[ \n\
                                      os.path.join('.' , 'src', 'gpu_regularisers.pyx' ),\n\
                                        ],\n\
                             include_dirs=extra_include_dirs, \n\
 							 library_dirs=extra_library_dirs, \n\
 							 extra_compile_args=extra_compile_args, \n\
 							 libraries=extra_libraries ), \n\
    ],\n\
 	zip_safe = False,	\n\
 	packages = {'ccpi','ccpi.filters'},\n\
)")
else()
  message("CUDA NOT FOUND")
  set(SETUP_GPU_WRAPPERS "#CUDA NOT FOUND")
endif()

configure_file("setup-regularisers.py.in" "setup-regularisers.py")

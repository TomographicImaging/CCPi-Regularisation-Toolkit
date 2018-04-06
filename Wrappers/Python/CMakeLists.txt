#   Copyright 2018 Edoardo Pasca
cmake_minimum_required (VERSION 3.0)

project(RegularizerPython)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.

set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)

# conda orchestrated build
message("CIL_VERSION ${CIL_VERSION}")
#include (GenerateExportHeader)

find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
  if (PYTHON_VERSION_MAJOR EQUAL "3")
    set (BOOST_PYTHON "python3")
	set (BOOST_NUMPY "numpy3")
  else()
    set (BOOST_PYTHON "python")
	set (BOOST_NUMPY "numpy")
  endif()
endif()

	
## Build the regularizers package as a library
message("Creating Regularizers as shared library")

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /DBOOST_ALL_NO_LIB /openmp /DCCPiCore_EXPORTS")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES 
        #${Boost_LIBRARIES}
		#"tiff"
		)
		
  message("library lib: ${LIBRARY_LIB}")
  
elseif(UNIX)
   set (FLAGS "-fopenmp -O2 -funsigned-char -Wall  -Wl,--no-undefined  -DCCPiReconstructionIterative_EXPORTS -std=c++0x")  
   set (EXTRA_LIBRARIES 
		#${Boost_LIBRARIES}
		#"tiff"
		"gomp"
		)
endif()

# GPU Regularizers

find_package(CUDA)
if (CUDA_FOUND)
  message("CUDA FOUND")
  set (SETUP_GPU_WRAPPERS "extra_libraries += ['cilregcuda']\n\
setup( \n\
    name='ccpi', \n\
 	description='CCPi Core Imaging Library - Image Regularizers GPU',\n\
 	version=cil_version,\n\
    cmdclass = {'build_ext': build_ext},\n\
    ext_modules = [Extension('ccpi.filters.gpu_regularizers',\n\
                              sources=[ \n\
                                      os.path.join('.' , 'src', 'gpu_regularizers.pyx' ),\n\
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

configure_file("setup-regularizers.py.in" "setup-regularizers.py")


#add_executable(regularizer_test ${CMAKE_CURRENT_SOURCE_DIR}/test/test_regularizer.cpp)

#target_link_libraries (regularizer_test LINK_PUBLIC regularizers_lib)

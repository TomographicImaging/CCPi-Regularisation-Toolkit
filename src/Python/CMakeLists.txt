#   Copyright 2018 Edoardo Pasca
cmake_minimum_required (VERSION 3.0)

project(regulariserPython)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.

#set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)
message("Creating Python Wrapper")
# conda orchestrated build
message("CIL_VERSION: ${CIL_VERSION}")
#include (GenerateExportHeader)

find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
endif()

	
## Build the regularisers package as a library
#TODO message("Creating Regularisers as shared library")

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /openmp /DCCPiCore_EXPORTS")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES)
		
  message("library lib: ${LIBRARY_LIB}")
endif()

# GPU regularisers
if (BUILD_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
      message("CUDA FOUND")
      set(GPU_EXTRA_LIBRARIES "cilregcuda")
      set(SETUP_GPU_WRAPPERS "Extension('ccpi.filters.gpu_regularisers',\n\
                                  sources=[ \n\
                                          os.path.join('.' , 'src', 'gpu_regularisers.pyx' ),\n\
                                            ],\n\
                                 include_dirs=extra_include_dirs, \n\
                                 library_dirs=extra_library_dirs, \n\
                                 extra_compile_args=extra_compile_args, \n\
                                 libraries=extra_libraries ), \n
      ")
      
    else()
      message("CUDA NOT FOUND")
      set(SETUP_GPU_WRAPPERS "#CUDA NOT FOUND")
      set(GPU_EXTRA_LIBRARIES "")
    endif()
else()
set(SETUP_GPU_WRAPPERS "#CUDA NOT BUILT")
set(GPU_EXTRA_LIBRARIES "")
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup-regularisers.py.in" "${CMAKE_CURRENT_BINARY_DIR}/setup-regularisers.py")


find_package(PythonInterp)
find_package(PythonLibs)
if (PYTHONINTERP_FOUND)
  message(STATUS "Found PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
  message(STATUS "Python version ${PYTHON_VERSION_STRING}")
endif()
if (PYTHONLIBS_FOUND)
  message(STATUS "Found PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
  message(STATUS "Found PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")
endif()

if (PYTHONINTERP_FOUND)
    message("Python found " ${PYTHON_EXECUTABLE})
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup-regularisers.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    #set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/module/__init__.py")
    set (DEPS       "${CMAKE_BINARY_DIR}/src/Core/")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    message("Core binary dir " ${CMAKE_BINARY_DIR}/Core/${CMAKE_BUILD_TYPE})
    
    if (CONDA_BUILD)
      set (LIBRARY_LIB ${CMAKE_BINARY_DIR}/src/Core)
      set (PYTHON_ARGS install)
    elseif (WIN32)
      set (LIBRARY_LIB ${CMAKE_BINARY_DIR}/src/Core/${CMAKE_BUILD_TYPE})
      set (PYTHON_ARGS build_ext --inplace)
    else()
      set (LIBRARY_LIB ${CMAKE_BINARY_DIR}/src/Core)
      set (PYTHON_ARGS build_ext --inplace)
    endif()

    add_custom_command(OUTPUT ${OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ccpi ${CMAKE_CURRENT_BINARY_DIR}/ccpi
      COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
        PREFIX=${CMAKE_SOURCE_DIR}/src/Core
        LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
        LIBRARY_LIB=${LIBRARY_LIB}
        ${PYTHON_EXECUTABLE} -m pip install . --verbose
      COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
      DEPENDS cilreg)

    if (NOT CONDA_BUILD)
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ccpi 
              DESTINATION ${PYTHON_DEST})
    endif()
    
    add_custom_target(PythonWrapper ALL DEPENDS ${OUTPUT})

    add_custom_target(copy-runtime-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../Core/regularisers_GPU/cuda_kernels ${CMAKE_CURRENT_BINARY_DIR}/ccpi/cuda_kernels)     
endif()

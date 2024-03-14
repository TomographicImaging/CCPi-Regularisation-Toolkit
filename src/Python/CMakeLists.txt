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

  message("Python found " ${PYTHON_EXECUTABLE})
  # set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
  # add_custom_command(OUTPUT ${OUTPUT}
  #     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
  #     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ccpi ${CMAKE_CURRENT_BINARY_DIR}/ccpi
  #     COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
  #       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
  #       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
  #       LIBRARY_LIB=${LIBRARY_LIB}
  #       ${PYTHON_EXECUTABLE} -m pip install . --verbose
  #     COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
  #     DEPENDS cilreg)
  
  # add_custom_target(PythonWrapper ALL DEPENDS ${OUTPUT})

  add_custom_target(PythonWrapper ALL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON_EXECUTABLE} -m pip install . --verbose)
    
  add_custom_target(copy-runtime-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../Core/regularisers_GPU/cuda_kernels ${CMAKE_CURRENT_BINARY_DIR}/ccpi/cuda_kernels)     
endif()

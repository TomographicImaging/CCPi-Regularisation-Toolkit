cmake_minimum_required(VERSION 3.5)

project(RGL)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Options: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# The version number.
set (RGL_VERSION_MAJOR 1)
set (RGL_VERSION_MINOR 0)

set (CIL_VERSION_MAJOR 0)
set (CIL_VERSION_MINOR 10)
set (CIL_VERSION_PATCH 2)

# set (CIL_VERSION '${CIL_VERSION_MAJOR}.${CIL_VERSION_MINOR}.${CIL_VERSION_PATCH}' CACHE INTERNAL "Core Imaging Library version" FORCE)
# get CIL_VERSION from environment variable and set it in cmake
set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)
message(STATUS "CIL_VERSION: ${CIL_VERSION}")

# set the Python variables for the Conda environment
#include(${CMAKE_SOURCE_DIR}/CMake/FindAnacondaEnvironment.cmake)

option (BUILD_MATLAB_WRAPPER "Build Matlab Wrappers" OFF)
option (BUILD_PYTHON_WRAPPER "Build Python Wrappers" ON)
option (CONDA_BUILD "Conda Build" OFF)
option (BUILD_CUDA "Build the CUDA modules" ON)

set(MATLAB_DEST_DIR "" CACHE PATH "Directory of the Matlab wrappers")
if (MATLAB_DEST_DIR)
set(MATLAB_DEST "${MATLAB_DEST_DIR}")
else()
set(MATLAB_DEST "${CMAKE_INSTALL_PREFIX}/matlab")
endif()
message(STATUS "Matlab wrappers will be installed in " ${MATLAB_DEST})

set(PYTHON_DEST_DIR "" CACHE PATH "Directory of the Python wrappers")
if (PYTHON_DEST_DIR)
 set(PYTHON_DEST "${PYTHON_DEST_DIR}")
else()
 set(PYTHON_DEST "${CMAKE_INSTALL_PREFIX}/python")
endif()
message(STATUS "Python wrappers will be installed in " ${PYTHON_DEST})


#add_subdirectory(src/Core)
add_subdirectory(src)

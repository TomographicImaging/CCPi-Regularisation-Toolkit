cmake_minimum_required(VERSION 3.23)

project(RGL)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Options: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# set the Python variables for the Conda environment
#include(${CMAKE_SOURCE_DIR}/CMake/FindAnacondaEnvironment.cmake)

option (BUILD_MATLAB_WRAPPER "Build Matlab Wrappers" OFF)
option (BUILD_PYTHON_WRAPPER "Build Python Wrappers" ON)
option (BUILD_CUDA "Build the CUDA modules" ON)

set(MATLAB_DEST_DIR "" CACHE PATH "Directory of the Matlab wrappers")
if (MATLAB_DEST_DIR)
set(MATLAB_DEST "${MATLAB_DEST_DIR}")
else()
set(MATLAB_DEST "${CMAKE_INSTALL_PREFIX}/matlab")
endif()
message(STATUS "Matlab wrappers will be installed in " ${MATLAB_DEST})

#add_subdirectory(src/Core)
add_subdirectory(src)

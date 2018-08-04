#   Copyright 2017 Edoardo Pasca
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

cmake_minimum_required (VERSION 3.0)

project(RGL)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.
set (RGL_VERSION_MAJOR 1)
set (RGL_VERSION_MINOR 0)

set (CIL_VERSION_MAJOR 0)
set (CIL_VERSION_MINOR 10)
set (CIL_VERSION_PATCH 1)

set (CIL_VERSION '${CIL_VERSION_MAJOR}.${CIL_VERSION_MINOR}.${CIL_VERSION_PATCH}' CACHE INTERNAL "Core Imaging Library version" FORCE)

# set the Python variables for the Conda environment
#include(${CMAKE_SOURCE_DIR}/CMake/FindAnacondaEnvironment.cmake)

add_subdirectory(Core)
if (BUILD_WRAPPERS)
  add_subdirectory(Wrappers)
endif()
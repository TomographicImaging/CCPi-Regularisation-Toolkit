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

message("CIL VERSION " ${CIL_VERSION})

# variables that are set by conda
#PREFIX=C:\Apps\Miniconda2\envs\cil\Library
#LIBRARY_INC=C:\\Apps\\Miniconda2\\envs\\cil\\Library\\include

set (NUMPY_VERSION 1.12)
#set (PYTHON_VERSION 3.5)

#https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/R9gWjl09UFs
set (CONDA_ENVIRONMENT "cil")
set (CONDA_ENVIRONMENT_PATH "C:\\Apps\\Miniconda2\\envs\\${CONDA_ENVIRONMENT}" CACHE PATH "env dir")

message("CIL VERSION " ${CIL_VERSION})

# set the Python variables for the Conda environment
include(FindAnacondaEnvironment.cmake)
findPythonForAnacondaEnvironment(${CONDA_ENVIRONMENT_PATH})
message("Python found " ${PYTHON_VERSION_STRING})
findPythonPackagesPath()
message("PYTHON_PACKAGES_FOUND " ${PYTHON_PACKAGES_PATH})

# copy the Pyhon files of the package
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ccpi/imaging/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/imaging/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi/imaging)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/imaging/Regularizer.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi/imaging)


# Copy and configure the relative conda build and recipes
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conda-recipe)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conda-recipe/meta.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/conda-recipe)

if (WIN32)
	file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conda-recipe/bld.bat DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/conda-recipe/)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile.bat.in ${CMAKE_CURRENT_BINARY_DIR}/compile.bat)
elseif(UNIX)
	file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conda-recipe/build.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/conda-recipe/)
    # assumes we will use bash
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile.sh.in ${CMAKE_CURRENT_BINARY_DIR}/compile.sh)
endif()



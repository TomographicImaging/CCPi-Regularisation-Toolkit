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

# variables that must be set for conda compilation

#PREFIX=C:\Apps\Miniconda2\envs\cil\Library
#LIBRARY_INC=C:\\Apps\\Miniconda2\\envs\\cil\\Library\\include
set (NUMPY_VERSION 1.12)

## Tries to parse the output of conda env list to determine the current
## active conda environment
message ("Trying to determine your active conda environment...")
execute_process(COMMAND "conda" "env" "list"
					OUTPUT_VARIABLE _CONDA_ENVS
					RESULT_VARIABLE _CONDA_RESULT
					ERROR_VARIABLE _CONDA_ERR)
			if(NOT _CONDA_RESULT)
				string(REPLACE "\n" ";" ENV_LIST ${_CONDA_ENVS})
				foreach(line ${ENV_LIST})
				  string(REGEX MATCHALL "(.+)[*](.+)" match ${line})
				  if (NOT ${match} EQUAL "")
				    #message("MATCHED " ${CMAKE_MATCH_0})
					#message("MATCHED " ${CMAKE_MATCH_1})
					#message("MATCHED " ${CMAKE_MATCH_2})
					string(STRIP ${CMAKE_MATCH_1} CONDA_ENVIRONMENT)
					string(STRIP ${CMAKE_MATCH_2} CONDA_ENVIRONMENT_PATH)   
				  endif()
				endforeach()
			else()
				message(FATAL_ERROR "ERROR with conda command " ${_CONDA_ERR})
			endif()

if (${CONDA_ENVIRONMENT} AND ${CONDA_ENVIRONMENT_PATH})
  message (FATAL_ERROR "CONDA NOT FOUND")
else()
  message("**********************************************************")
  message("Using current conda environmnet " ${CONDA_ENVIRONMENT})
  message("Using current conda environmnet path " ${CONDA_ENVIRONMENT_PATH})
endif()

message("CIL VERSION " ${CIL_VERSION})

# set the Python variables for the Conda environment
include(FindAnacondaEnvironment.cmake)
findPythonForAnacondaEnvironment(${CONDA_ENVIRONMENT_PATH})

message("Python found " ${PYTHON_VERSION_STRING})
message("Python found Major " ${PYTHON_VERSION_MAJOR})
message("Python found Minor " ${PYTHON_VERSION_MINOR})

findPythonPackagesPath()
message("PYTHON_PACKAGES_FOUND " ${PYTHON_PACKAGES_PATH})

## CACHE SOME VARIABLES ##
set (CONDA_ENVIRONMENT ${CONDA_ENVIRONMENT} CACHE INTERNAL "active conda environment" FORCE)
set (CONDA_ENVIRONMENT_PATH ${CONDA_ENVIRONMENT_PATH} CACHE INTERNAL "active conda environment" FORCE)

set (PYTHON_VERSION_STRING ${PYTHON_VERSION_STRING} CACHE INTERNAL "conda environment Python version string" FORCE)
set (PYTHON_VERSION_MAJOR ${PYTHON_VERSION_MAJOR} CACHE INTERNAL "conda environment Python version major" FORCE)
set (PYTHON_VERSION_MINOR ${PYTHON_VERSION_MINOR} CACHE INTERNAL "conda environment Python version minor" FORCE)
set (PYTHON_VERSION_PATCH ${PYTHON_VERSION_PATCH} CACHE INTERNAL "conda environment Python version patch" FORCE)
set (PYTHON_PACKAGES_PATH ${PYTHON_PACKAGES_PATH} CACHE INTERNAL "conda environment Python packages path" FORCE)

if (WIN32)
  #set (CONDA_ENVIRONMENT_PATH "C:\\Apps\\Miniconda2\\envs\\${CONDA_ENVIRONMENT}" CACHE PATH "Main environment directory")
  set (CONDA_ENVIRONMENT_PREFIX "${CONDA_ENVIRONMENT_PATH}\\Library" CACHE PATH "env dir")
  set (CONDA_ENVIRONMENT_LIBRARY_INC "${CONDA_ENVIRONMENT_PREFIX}\\include" CACHE PATH "env dir")
elseif (UNIX)
  #set (CONDA_ENVIRONMENT_PATH "/apps/anaconda/2.4/envs/${CONDA_ENVIRONMENT}" CACHE PATH "Main environment directory")
  set (CONDA_ENVIRONMENT_PREFIX "${CONDA_ENVIRONMENT_PATH}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" CACHE PATH "env dir")
  set (CONDA_ENVIRONMENT_LIBRARY_INC "${CONDA_ENVIRONMENT_PREFIX}/include" CACHE PATH "env dir")
endif()

######### CONFIGURE REGULARIZER PACKAGE #############

# copy the Pyhon files of the package regularizer
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ccpi/imaging/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi)
# regularizers
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
  
  message ("We are on UNIX")
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/conda-recipe/build.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/conda-recipe/)
  # assumes we will use bash
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile.sh.in ${CMAKE_CURRENT_BINARY_DIR}/compile.sh)

endif()

########## CONFIGURE FISTA RECONSTRUCTOR PACKAGE
# fista reconstructor
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/reconstruction/FISTAReconstructor.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi/reconstruction)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ccpi/reconstruction/__init__.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ccpi/reconstruction)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup-fista.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup-fista.py)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fista-recipe)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/fista-recipe/meta.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fista-recipe)

if (WIN32)

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/fista-recipe/bld.bat DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fista-recipe/)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile-fista.bat.in ${CMAKE_CURRENT_BINARY_DIR}/compile-fista.bat)

elseif(UNIX)
  message ("We are on UNIX")
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/fista-recipe/build.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fista-recipe/)
  # assumes we will use bash
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile-fista.sh.in ${CMAKE_CURRENT_BINARY_DIR}/compile-fista.sh)
endif()

#############################  TARGETS

##########################  REGULARIZER PACKAGE ###############################

# runs cmake on the build tree to update the code from source
add_custom_target(update_code 
		COMMAND ${CMAKE_COMMAND} 
		ARGS ${CMAKE_SOURCE_DIR}
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		) 


add_custom_target(fista
		COMMAND bash
		compile-fista.sh
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		DEPENDS ${update_code}
		)

add_custom_target(regularizers
		COMMAND bash
		compile.sh
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		DEPENDS update_code
		)

add_custom_target(install-fista
		COMMAND ${CONDA_EXECUTABLE}
		install --force --use-local ccpi-fista=${CIL_VERSION} -c ccpi -c conda-forge
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		)

add_custom_target(install-regularizers
		COMMAND ${CONDA_EXECUTABLE}
		install --force --use-local ccpi-regularizers=${CIL_VERSION} -c ccpi -c conda-forge
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		)
### add tests

#add_executable(RegularizersTest )
#find_package(tiff)
#if (TIFF_FOUND)
#  message("LibTIFF Found")
#  message("TIFF_INCLUDE_DIR " ${TIFF_INCLUDE_DIR})
#  message("TIFF_LIBRARIES" ${TIFF_LIBRARIES})
#else()
#  message("LibTIFF not found")
#endif()

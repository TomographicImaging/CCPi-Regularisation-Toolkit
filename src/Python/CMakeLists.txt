cmake_minimum_required (VERSION 3.0)

project(FISTA)

# The version number.
set (FISTA_VERSION_MAJOR 1)
set (FISTA_VERSION_MINOR 0)

set (CIL_VERSION_MAJOR 0)
set (CIL_VERSION_MINOR 9)
set (CIL_VERSION_PATCH 1)

set (CIL_VERSION '${CIL_VERSION_MAJOR}.${CIL_VERSION_MINOR}.${CIL_VERSION_PATCH}')

message("CIL VERSION " ${CIL_VERSION})

# variables we need to run the conda build
#PREFIX=C:\Apps\Miniconda2\envs\cil\Library
#LIBRARY_INC=C:\\Apps\\Miniconda2\\envs\\cil\\Library\\include

set (NUMPY_VERSION 1.12)
#set (PYTHON_VERSION 3.5)

#https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/R9gWjl09UFs
set (CONDA_ENVIRONMENT "C:\\Apps\\Miniconda2\\envs\\cil27" CACHE PATH "env dir")

function (findPythonForAnacondaEnvironment env)
	
	file(TO_CMAKE_PATH ${env}/python.exe PYTHON_EXECUTABLE)
	
	message("Found " ${PYTHON_EXECUTABLE})
	execute_process(COMMAND ${PYTHON_EXECUTABLE} pythonversion.py major 
					WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
					OUTPUT_VARIABLE major
					ERROR_QUIET)
	execute_process(COMMAND ${PYTHON_EXECUTABLE} pythonversion.py minor 
					WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
					OUTPUT_VARIABLE minor
					 ERROR_QUIET)
	execute_process(COMMAND ${PYTHON_EXECUTABLE} pythonversion.py patch 
					WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
					OUTPUT_VARIABLE patch
					 ERROR_QUIET)
	execute_process(COMMAND ${PYTHON_EXECUTABLE} pythonversion.py  
					WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
					OUTPUT_VARIABLE version
					 ERROR_QUIET)

	
	set (PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
	set (PYTHONINTERP_FOUND "ON" PARENT_SCOPE)
	set (PYTHON_VERSION_STRING ${version})
	message("My version found " ${PYTHON_VERSION_STRING})

endfunction()

findPythonForAnacondaEnvironment(${CONDA_ENVIRONMENT})

set(Python_ADDITIONAL_VERSIONS 3)

find_package(PythonInterp)
if (PYTHONINTERP_FOUND)
  
  message("Found interpret " ${PYTHON_EXECUTABLE})
  
  foreach(pv ${PYTHON_VERSION_STRING})
    message("Found interpret " ${pv})
  endforeach()
endif()

find_package(PythonLibs)
if (PYTHONLIB_FOUND) 
  message("Found PythonLibs PYTHON_LIBRARIES " ${PYTHON_LIBRARIES})
  message("Found PythonLibs PYTHON_INCLUDE_PATH " ${PYTHON_INCLUDE_PATH})
  message("Found PythonLibs PYTHON_INCLUDE_DIRS " ${PYTHON_INCLUDE_DIRS})
  message("Found PythonLibs PYTHONLIBS_VERSION_STRING " ${PYTHONLIBS_VERSION_STRING}  )
  
endif()


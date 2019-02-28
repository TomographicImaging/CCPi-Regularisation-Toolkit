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

# #.rst:
# FindAnacondaEnvironment
# --------------
#
# Find Python executable and library for a specific Anaconda environment
#
# This module finds the Python interpreter for a specific Anaconda enviroment, 
# if installed and determines where the include files and libraries are.  
# This code sets the following variables:
#
# ::
#   PYTHONINTERP_FOUND         - if the Python interpret has been found
#   PYTHON_EXECUTABLE          - the Python interpret found
#   PYTHON_LIBRARY             - path to the python library
#   PYTHON_INCLUDE_PATH        - path to where Python.h is found (deprecated)
#   PYTHON_INCLUDE_DIRS        - path to where Python.h is found
#   PYTHONLIBS_VERSION_STRING  - version of the Python libs found (since CMake 2.8.8)
#   PYTHON_VERSION_MAJOR       - major Python version
#   PYTHON_VERSION_MINOR       - minor Python version
#   PYTHON_VERSION_PATCH       - patch Python version



function (findPythonForAnacondaEnvironment env)
	if (WIN32)
	  file(TO_CMAKE_PATH ${env}/python.exe PYTHON_EXECUTABLE)
        elseif (UNIX)
  	  file(TO_CMAKE_PATH ${env}/bin/python PYTHON_EXECUTABLE)
	endif()

	
	message("findPythonForAnacondaEnvironment Found Python Executable" ${PYTHON_EXECUTABLE})
	####### FROM FindPythonInterpr ########
	# determine python version string
	if(PYTHON_EXECUTABLE)
		execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
								"import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
						OUTPUT_VARIABLE _VERSION
						RESULT_VARIABLE _PYTHON_VERSION_RESULT
						ERROR_QUIET)
		if(NOT _PYTHON_VERSION_RESULT)
			string(REPLACE ";" "." _PYTHON_VERSION_STRING "${_VERSION}")
			list(GET _VERSION 0 _PYTHON_VERSION_MAJOR)
			list(GET _VERSION 1 _PYTHON_VERSION_MINOR)
			list(GET _VERSION 2 _PYTHON_VERSION_PATCH)
			if(PYTHON_VERSION_PATCH EQUAL 0)
				# it's called "Python 2.7", not "2.7.0"
				string(REGEX REPLACE "\\.0$" "" _PYTHON_VERSION_STRING "${PYTHON_VERSION_STRING}")
			endif()
		else()
			# sys.version predates sys.version_info, so use that
			execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; sys.stdout.write(sys.version)"
							OUTPUT_VARIABLE _VERSION
							RESULT_VARIABLE _PYTHON_VERSION_RESULT
							ERROR_QUIET)
			if(NOT _PYTHON_VERSION_RESULT)
				string(REGEX REPLACE " .*" "" _PYTHON_VERSION_STRING "${_VERSION}")
				string(REGEX REPLACE "^([0-9]+)\\.[0-9]+.*" "\\1" _PYTHON_VERSION_MAJOR "${PYTHON_VERSION_STRING}")
				string(REGEX REPLACE "^[0-9]+\\.([0-9])+.*" "\\1" _PYTHON_VERSION_MINOR "${PYTHON_VERSION_STRING}")
				if(PYTHON_VERSION_STRING MATCHES "^[0-9]+\\.[0-9]+\\.([0-9]+)")
					set(PYTHON_VERSION_PATCH "${CMAKE_MATCH_1}")
				else()
					set(PYTHON_VERSION_PATCH "0")
				endif()
			else()
				# sys.version was first documented for Python 1.5, so assume
				# this is older.
				set(PYTHON_VERSION_STRING "1.4" PARENT_SCOPE)
				set(PYTHON_VERSION_MAJOR "1" PARENT_SCOPE)
				set(PYTHON_VERSION_MINOR "4" PARENT_SCOPE)
				set(PYTHON_VERSION_PATCH "0" PARENT_SCOPE)
			endif()
		endif()
		unset(_PYTHON_VERSION_RESULT)
		unset(_VERSION)
	endif()
	###############################################
	
	set (PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
	set (PYTHONINTERP_FOUND "ON" PARENT_SCOPE)
	set (PYTHON_VERSION_STRING ${_PYTHON_VERSION_STRING} PARENT_SCOPE)
	set (PYTHON_VERSION_MAJOR ${_PYTHON_VERSION_MAJOR} PARENT_SCOPE)
	set (PYTHON_VERSION_MINOR ${_PYTHON_VERSION_MINOR} PARENT_SCOPE)
	set (PYTHON_VERSION_PATCH ${_PYTHON_VERSION_PATCH} PARENT_SCOPE)
	message("My version found " ${PYTHON_VERSION_STRING})
	## find conda executable
	if (WIN32)
	  set (CONDA_EXECUTABLE ${env}/Script/conda PARENT_SCOPE)
	elseif(UNIX)
	  set (CONDA_EXECUTABLE ${env}/bin/conda PARENT_SCOPE)
	endif()
endfunction()



set(Python_ADDITIONAL_VERSIONS 3.5)

find_package(PythonInterp)
if (PYTHONINTERP_FOUND)
  
  message("Found interpret " ${PYTHON_EXECUTABLE})
  message("Python Library " ${PYTHON_LIBRARY})
  message("Python Include Dir " ${PYTHON_INCLUDE_DIR})
  message("Python Include Path " ${PYTHON_INCLUDE_PATH})
  
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
else()
  message("No PythonLibs Found")  
endif()




function(findPythonPackagesPath)
   execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print (get_python_lib())"
                      RESULT_VARIABLE PYTHON_CVPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
   #message("STD_PACKAGES " ${PYTHON_STD_PACKAGES_PATH})
   if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
        set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/site-packages")
   endif()

    SET(PYTHON_PACKAGES_PATH "${PYTHON_STD_PACKAGES_PATH}" PARENT_SCOPE)

endfunction()



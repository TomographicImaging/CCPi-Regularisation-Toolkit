project(regulariserPython)

message(STATUS "Creating Python Wrapper")
#include (GenerateExportHeader)

cmake_policy(SET CMP0094 NEW)
set(Python_FIND_REGISTRY LAST) # prefer active over base conda env
set(Python_FIND_VIRTUALENV STANDARD) # use PATH to search
find_package(Python COMPONENTS Interpreter REQUIRED)

set(PIP_CMAKE_DEPS cilreg)
if(TARGET cilregcuda)
  list(APPEND PIP_CMAKE_DEPS cilregcuda)
endif()
if (UNIX)
  install(TARGETS ${PIP_CMAKE_DEPS}
	  LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/ccpi/filters"
	  CONFIGURATIONS ${CMAKE_BUILD_TYPE})
elseif(WIN32)
  install(TARGETS ${PIP_CMAKE_DEPS}
	  RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/ccpi/filters"
	  ARCHIVE DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/ccpi/filters"
	  CONFIGURATIONS ${CMAKE_BUILD_TYPE})
endif()

add_custom_target(PythonWrapper ALL
  COMMAND "${Python_EXECUTABLE}" -m pip install "${CMAKE_CURRENT_SOURCE_DIR}" --verbose
  DEPENDS ${PIP_CMAKE_DEPS})

add_custom_target(copy-runtime-files ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../Core/regularisers_GPU/cuda_kernels ${CMAKE_CURRENT_BINARY_DIR}/ccpi/cuda_kernels)

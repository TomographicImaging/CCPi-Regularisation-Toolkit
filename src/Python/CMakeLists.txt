project(regulariserPython)

message(STATUS "Creating Python Wrapper")
#include (GenerateExportHeader)

find_package(PythonInterp REQUIRED)
message(DEBUG "Current Python ${PYTHON_VERSION_STRING} found ${PYTHON_EXECUTABLE}")
message(DEBUG "Python found: ${PYTHON_EXECUTABLE}")
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

set(PIP_CMAKE_DEPS cilreg)
if(TARGET cilregcuda)
  list(APPEND PIP_CMAKE_DEPS cilregcuda)
endif()
add_custom_target(PythonWrapper ALL
  COMMAND "${PYTHON_EXECUTABLE}" -m pip install ${CMAKE_CURRENT_SOURCE_DIR} --verbose
  DEPENDS ${PIP_CMAKE_DEPS})

add_custom_target(copy-runtime-files ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../Core/regularisers_GPU/cuda_kernels ${CMAKE_CURRENT_BINARY_DIR}/ccpi/cuda_kernels)

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

# If APPLE don't use @rpath
if (APPLE)
  option(SHARED_LIBS_ABS_PATH "Force shared libraries to be installed with absolute paths (as opposed to rpaths)" ON)
  mark_as_advanced( SHARED_LIBS_ABS_PATH )  
  if (SHARED_LIBS_ABS_PATH)
    # Set install_name_dir as the absolute path to install_prefix/lib
    GET_FILENAME_COMPONENT(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib REALPATH)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif(SHARED_LIBS_ABS_PATH)
endif(APPLE)

find_package(OpenMP REQUIRED)
# add_definitions(${OpenMP_C_FLAGS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0") 
  set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_C)
else()
  message(WARNING "Your CMake version is old. OpenMP linking flags  might be incorrect.")
  # need to explicitly set this. Definitely for gcc, hopefully also for other systems.
  # See https://gitlab.kitware.com/cmake/cmake/issues/15392
  set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
  set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
endif()

add_subdirectory(Core)
if (BUILD_MATLAB_WRAPPER)
    add_subdirectory(Matlab)
endif()
if (BUILD_PYTHON_WRAPPER)
    add_subdirectory(Python)
endif()
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

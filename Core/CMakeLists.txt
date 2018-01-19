#   Copyright 2018 Edoardo Pasca
cmake_minimum_required (VERSION 3.0)

project(RegularizerLibrary)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.

set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)

# conda orchestrated build
message("CIL_VERSION ${CIL_VERSION}")

find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
  if (PYTHON_VERSION_MAJOR EQUAL "3")
    set (BOOST_PYTHON "python3")
	set (BOOST_NUMPY "numpy3")
  else()
    set (BOOST_PYTHON "python")
	set (BOOST_NUMPY "numpy")
  endif()
endif()

find_package(Boost  REQUIRED
	COMPONENTS ${BOOST_PYTHON} ${BOOST_NUMPY})

if (Boost_FOUND)
   message("Boost version " ${Boost_VERSION})
   message("Boost include dir " ${Boost_INCLUDE_DIRS})
   message("Boost library dir " ${Boost_LIBRARY_DIRS})
   message("Boost libraries " ${Boost_LIBRARIES})
endif()
	
## Build the regularizers package as a library
message("Creating Iterative Reconstruction CGLS as shared library")
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

if(WIN32)
 set (FLAGS "/DWIN32 /EHsc /DBOOST_ALL_NO_LIB /openmp  /DCCPiReconstructionIterative_EXPORTS")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  
  set (EXTRA_LIBRARIES 
        ${Boost_LIBRARIES}
		#"tiff"
		)
		
  message("library lib: ${LIBRARY_LIB}")
  
elseif(UNIX)
   set (FLAGS "-fopenmp -O2 -funsigned-char -Wall  -Wl,--no-undefined  -DCCPiReconstructionIterative_EXPORTS -std=c++0x")  
   set (EXTRA_LIBRARIES 
		${Boost_LIBRARIES}
		#"tiff"
		"gomp"
		)
   
endif()
## Build the regularizers package as a library
message("Adding regularizers as shared library")

#set(CMAKE_C_COMPILER /apps/pgi/linux86-64/17.4/bin/pgcc)
#set(CMAKE_C_FLAGS "-acc -Minfo -ta=tesla:cc20 -openmp")
#set(CMAKE_C_FLAGS "-acc -Minfo -ta=multicore -openmp -fPIC")
add_library(cilreg SHARED
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/FGP_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/LLT_model_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/PatchBased_Regul_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/SplitBregman_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/TGV_PD_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/utils.c
	    )

if (UNIX)
message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS cilreg 
	LIBRARY DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
elseif(WIN32)
message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
  install(TARGETS cilreg 
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)
endif()

#add_executable(regularizer_test ${CMAKE_CURRENT_SOURCE_DIR}/test/test_regularizer.cpp)

#target_link_libraries (regularizer_test LINK_PUBLIC regularizers_lib)
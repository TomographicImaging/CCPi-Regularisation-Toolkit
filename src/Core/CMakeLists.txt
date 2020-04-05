#   Copyright 2018 Edoardo Pasca
#cmake_minimum_required (VERSION 3.0)

project(RGL_core)
#https://stackoverflow.com/questions/13298504/using-cmake-with-setup-py

# The version number.

set (CIL_VERSION $ENV{CIL_VERSION} CACHE INTERNAL "Core Imaging Library version" FORCE)

# conda orchestrated build
message("CIL_VERSION ${CIL_VERSION}")
#include (GenerateExportHeader)

## Build the regularisers package as a library
message("Creating Regularisers as a shared library")

find_package(GLIB2 REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -fno-omit-frame-pointer")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

set (EXTRA_LIBRARIES "")
if(WIN32)
  set (FLAGS "/DWIN32 /EHsc /DCCPiCore_EXPORTS /openmp")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT.lib")
  message("library lib: ${LIBRARY_LIB}")
elseif(APPLE)
  set (FLAGS "-DCCPiReconstructionIterative_EXPORTS ")
elseif(UNIX)
   set (FLAGS "-O2 -funsigned-char -Wall  -Wl,--no-undefined  -DCCPiReconstructionIterative_EXPORTS ")
   set(EXTRA_LIBRARIES "m")
endif()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

## Build the regularisers package as a library
message("Adding regularisers as a shared library")

#set(CMAKE_C_COMPILER /opt/intel/compilers_and_libraries/linux/bin/intel64/icc)
#set(CMAKE_C_FLAGS "-Ofast -mtune=sandybridge -xSSE4.2 -qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=vec -qopenmp")
#set(CMAKE_C_FLAGS "-Ofast -mtune=sandybridge -axAVX2 -xAVX2 -qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=vec -qopenmp")
#set(CMAKE_C_FLAGS "-Ofast -mtune=sandybridge -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl -qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=vec -qopenmp")

#set(CMAKE_C_COMPILER clang)
#set(CMAKE_C_FLAGS "-march=nocona -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -ftree-vectorize -fopenmp")

#set(CMAKE_C_COMPILER gcc-9)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec -fopenmp")
#set(CMAKE_C_FLAGS "-march=nocona -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec -mprefer-vector-width=128 -fopenmp")
#set(CMAKE_C_FLAGS "-march=native -mavx2 -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec -mprefer-vector-width=512 -fopenmp")
#set(CMAKE_C_FLAGS "-march=native -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec -mprefer-vector-width=512 -fopenmp")
#set(CMAKE_C_FLAGS_RELEASE "-g -gdwarf-2 -g3 -fno-omit-frame-pointer")

#set(CMAKE_C_FLAGS "-acc -Minfo -ta=tesla:cc20 -openmp")
#set(CMAKE_C_FLAGS "-acc -Minfo -ta=multicore -openmp -fPIC")
add_library(cilreg SHARED
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/hw_sched.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/hw_thread.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/FGP_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/SB_TV_core.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/PD_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/TGV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/Diffusion_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/Diffus4th_order_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/LLT_ROF_core.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/ROF_TV_core.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/FGP_dTV_core.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/TNV_core.c
            #${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/TNV_core_fast.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/Nonlocal_TV_core.c
            ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/PatchSelect_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/utils.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/inpainters_CPU/Diffusion_Inpaint_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/inpainters_CPU/NonlocalMarching_Inpaint_core.c
	    )
target_link_libraries(cilreg ${EXTRA_LIBRARIES} ${GLIB2_LIBRARIES} ${GTHREAD2_LIBRARIES} )
include_directories(cilreg PUBLIC
                      ${GLIB2_INCLUDE_DIR}
                      ${LIBRARY_INC}/include
					            ${CMAKE_CURRENT_SOURCE_DIR}
		                  ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_CPU/
		                  ${CMAKE_CURRENT_SOURCE_DIR}/inpainters_CPU/  )

## Install

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



# GPU Regularisers
if (BUILD_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
      set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC -shared -D_FORCE_INLINES")
      message("CUDA FLAGS ${CUDA_NVCC_FLAGS}")
      CUDA_ADD_LIBRARY(cilregcuda SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/TV_ROF_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/TV_FGP_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/TV_PD_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/TV_SB_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/LLT_ROF_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/TGV_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/dTV_FGP_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/NonlDiff_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/Diffus_4thO_GPU_core.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/regularisers_GPU/PatchSelect_GPU_core.cu
      )
      if (UNIX)
        message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
        install(TARGETS cilregcuda
        LIBRARY DESTINATION lib
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
        )
      elseif(WIN32)
        message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
        install(TARGETS cilregcuda
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
        )
      endif()
    else()
      message("CUDA NOT FOUND")
    endif()
endif()

if (${BUILD_MATLAB_WRAPPER})
  if (WIN32)
        install(TARGETS cilreg DESTINATION ${MATLAB_DEST})
        if (CUDA_FOUND)
            install(TARGETS cilregcuda DESTINATION ${MATLAB_DEST})
        endif()
  endif()
endif()

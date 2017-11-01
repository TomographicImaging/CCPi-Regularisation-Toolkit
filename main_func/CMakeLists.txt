#   Copyright 2017 Edoardo Pasca
cmake_minimum_required (VERSION 3.0)

project(regularizers)

## Build the regularizers package as a library
message("Adding regularizers as shared library")
set(CMAKE_C_FLAGS "-fopenmp -O2 -funsigned-char -Wall -std=c++0x")
#set(CMAKE_C_COMPILER /apps/pgi/linux86-64/17.4/bin/pgcc)
#set(CMAKE_C_FLAGS "-acc -Minfo -ta=tesla:cc20 -openmp")
#set(CMAKE_C_FLAGS "-acc -Minfo -ta=multicore -openmp -fPIC")
add_library(regularizers_lib SHARED
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/FGP_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/LLT_model_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/PatchBased_Regul_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/SplitBregman_TV_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/TGV_PD_core.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/regularizers_CPU/utils.c
	    )
#GENERATE_EXPORT_HEADER(regularizers_lib)
install (TARGETS regularizers_lib DESTINATION $ENV{PREFIX}/lib)


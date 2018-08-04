project(regulariserMatlab)

message('<<<<<<<<<<<Compiling CPU regularisers>>>>>>>>>>>>>')

message('Compiling ROF-TV...')
#mex ROF_TV.c ROF_TV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
#movefile('ROF_TV.mex*',Pathmove);


find_package(Matlab COMPONENTS MAIN_PROGRAM MX_LIBRARY ENG_LIBRARY )
message ("Matlab mex library " ${Matlab_LIBRARIES})



#C:\Users\ofn77899\Documents\Projects\CCPi\GitHub\CCPi-FISTA_Reconstruction\Core\regularisers_CPU
matlab_add_mex(
    NAME MatlabWrapper
    SRC 
      #${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_CPU/ROF_TV.c 
      ${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_CPU/TNV.c 
    LINK_TO cilreg ${Matlab_LIBRARIES}
    )
    
target_include_directories(MatlabWrapper 
   PUBLIC ${CMAKE_SOURCE_DIR}/Core/regularisers_CPU
   ${CMAKE_SOURCE_DIR}/Core/regularisers_GPU
   ${CMAKE_SOURCE_DIR}/Core/inpainters_CPU
   ${CMAKE_SOURCE_DIR}/Core/
   ${MATLAB_INCLUDE_DIR})
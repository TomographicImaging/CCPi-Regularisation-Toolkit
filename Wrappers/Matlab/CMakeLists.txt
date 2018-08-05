project(regulariserMatlab)


find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM MX_LIBRARY ENG_LIBRARY )


#C:\Users\ofn77899\Documents\Projects\CCPi\GitHub\CCPi-FISTA_Reconstruction\Core\regularisers_CPU
# matlab_add_mex(
    # NAME CPU_ROF
    # SRC 
      # ${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_CPU/ROF_TV.c
    # LINK_TO cilreg ${Matlab_LIBRARIES}
    # )
    
# target_include_directories(CPU_ROF 
   # PUBLIC ${CMAKE_SOURCE_DIR}/Core/regularisers_CPU
   # ${CMAKE_SOURCE_DIR}/Core/regularisers_GPU
   # ${CMAKE_SOURCE_DIR}/Core/inpainters_CPU
   # ${CMAKE_SOURCE_DIR}/Core/
   # ${MATLAB_INCLUDE_DIR})
   
   # matlab_add_mex(
    # NAME CPU_TNV
    # SRC 
      # ${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_CPU/TNV.c 
    # LINK_TO cilreg ${Matlab_LIBRARIES}
    # )
    
# target_include_directories(CPU_TNV 
   # PUBLIC ${CMAKE_SOURCE_DIR}/Core/regularisers_CPU
   # ${CMAKE_SOURCE_DIR}/Core/regularisers_GPU
   # ${CMAKE_SOURCE_DIR}/Core/inpainters_CPU
   # ${CMAKE_SOURCE_DIR}/Core/
   # ${MATLAB_INCLUDE_DIR})
   
#set (CPU_MEX_FILES "regularisers_CPU/TNV.c;regularisers_CPU/ROF_TV.c")
#set (MEX_TARGETS "CPU_TNV;CPU_ROF")
#list(APPEND MEX_TARGETS "CPU_TNV")
#list(APPEND MEX_TARGETS "CPU_ROF")

set (CPU_TARGETS "CPU_TNV;CPU_ROF")
file(GLOB CPU_MEX_FILES
    "${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_CPU/*.c"
    "${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/inpainters_CPU/*.c"
)

list(LENGTH CPU_MEX_FILES num)


MATH(EXPR num "${num}-1")
#set(num "-1")

foreach(tgt RANGE ${num})
  message("number " ${tgt})
  #list(GET CPU_TARGETS ${tgt} current_target)
  list(GET CPU_MEX_FILES ${tgt} current_file)
    get_filename_component(current_target ${current_file} NAME)
  message("matlab_add_mex " ${current_target})
  matlab_add_mex(
    NAME ${current_target}
    SRC 
      ${current_file} 
    LINK_TO cilreg ${Matlab_LIBRARIES}
    )
    
target_include_directories(${current_target}
   PUBLIC ${CMAKE_SOURCE_DIR}/Core/regularisers_CPU
   ${CMAKE_SOURCE_DIR}/Core/regularisers_GPU
   ${CMAKE_SOURCE_DIR}/Core/inpainters_CPU
   ${CMAKE_SOURCE_DIR}/Core/
   ${MATLAB_INCLUDE_DIR})
   
   list(APPEND CPU_MEX_TARGETS ${current_target})
endforeach()
   
add_custom_target(MatlabWrapper DEPENDS ${CPU_MEX_TARGETS})

find_package(CUDA)
if (CUDA_FOUND)
  file(GLOB GPU_MEX_FILES
    "${CMAKE_SOURCE_DIR}/Wrappers/Matlab/mex_compile/regularisers_GPU/*.c"
  )

  list(LENGTH GPU_MEX_FILES num)


  MATH(EXPR num "${num}-1")
#set(num "-1")

  foreach(tgt RANGE ${num})
    message("number " ${tgt})
    #list(GET CPU_TARGETS ${tgt} current_target)
    list(GET GPU_MEX_FILES ${tgt} current_file)
    get_filename_component(current_target ${current_file} NAME)
    message("matlab_add_mex " ${current_target})
    matlab_add_mex(
      NAME ${current_target}
      SRC 
        ${current_file} 
      LINK_TO cilreg ${Matlab_LIBRARIES}
      )
    
    target_include_directories(${current_target}
    PUBLIC ${CMAKE_SOURCE_DIR}/Core/regularisers_CPU
           ${CMAKE_SOURCE_DIR}/Core/regularisers_GPU
           ${CMAKE_SOURCE_DIR}/Core/inpainters_CPU
           ${CMAKE_SOURCE_DIR}/Core/
           ${MATLAB_INCLUDE_DIR})
   
    list(APPEND CPU_MEX_TARGETS ${current_target})
  endforeach()
   
  add_custom_target(MatlabWrapperGPU DEPENDS ${GPU_MEX_TARGETS})

endif()

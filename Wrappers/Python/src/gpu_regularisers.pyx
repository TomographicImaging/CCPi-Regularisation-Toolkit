# distutils: language=c++
"""
Copyright 2018 CCPi
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Edoardo Pasca, Daniil Kazantsev
"""

import cython
import numpy as np
cimport numpy as np

cdef extern void TV_ROF_GPU_main(float* Input, float* Output, float lambdaPar, int iter, float tau, int N, int M, int Z);
cdef extern void TV_FGP_GPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int nonneg, int printM, int N, int M, int Z);

# Total-variation Rudin-Osher-Fatemi (ROF)
def TV_ROF_GPU(inputData,
                     regularisation_parameter,
                     iterations, 
                     time_marching_parameter):
    if inputData.ndim == 2:
        return ROFTV2D(inputData, 
                     regularisation_parameter,
                     iterations,
                     time_marching_parameter)
    elif inputData.ndim == 3:
        return ROFTV3D(inputData, 
                     regularisation_parameter,
                     iterations, 
                     time_marching_parameter)
                     
# Total-variation Fast-Gradient-Projection (FGP)
def TV_FGP_GPU(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     nonneg,
                     printM):
    if inputData.ndim == 2:
        return FGPTV2D(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     nonneg,
                     printM)
    elif inputData.ndim == 3:
        return FGPTV3D(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     nonneg,
                     printM)
                     
#****************************************************************#
#********************** Total-variation ROF *********************#
#****************************************************************#
def ROFTV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float time_marching_parameter):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    TV_ROF_GPU_main(            
            &inputData[0,0], &outputData[0,0], 
                       regularisation_parameter,
                       iterations , 
                       time_marching_parameter, 
                       dims[0], dims[1], 1);   
     
    return outputData
    
def ROFTV3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float time_marching_parameter):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
          
    # Running CUDA code here    
    TV_ROF_GPU_main(            
            &inputData[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter,
                       iterations , 
                       time_marching_parameter, 
                       dims[0], dims[1], dims[2]);   
     
    return outputData
#****************************************************************#
#********************** Total-variation FGP *********************#
#****************************************************************#
#******** Total-variation Fast-Gradient-Projection (FGP)*********#
def FGPTV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     int methodTV,
                     int nonneg,
                     int printM):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    TV_FGP_GPU_main(&inputData[0,0], &outputData[0,0],                        
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[0], dims[1], 1);   
     
    return outputData
    
def FGPTV3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     int methodTV,
                     int nonneg,
                     int printM):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
          
    # Running CUDA code here    
    TV_FGP_GPU_main(            
            &inputData[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[0], dims[1], dims[2]);   
     
    return outputData    

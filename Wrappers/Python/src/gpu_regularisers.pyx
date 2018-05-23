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
cdef extern void TV_SB_GPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int printM, int N, int M, int Z);
cdef extern void TGV_GPU_main(float *Input, float *Output, float lambdaPar, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY);
cdef extern void NonlDiff_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int N, int M, int Z);
cdef extern void dTV_FGP_GPU_main(float *Input, float *InputRef, float *Output, float lambdaPar, int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int printM, int N, int M, int Z);
cdef extern void Diffus4th_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int N, int M, int Z);

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
# Total-variation Split Bregman (SB)
def TV_SB_GPU(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     printM):
    if inputData.ndim == 2:
        return SBTV2D(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     printM)
    elif inputData.ndim == 3:
        return SBTV3D(inputData,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     methodTV,
                     printM)
# Total Generilised Variation (TGV)
def TGV_GPU(inputData, regularisation_parameter, alpha1, alpha0, iterations, LipshitzConst):
    if inputData.ndim == 2:
        return TGV2D(inputData, regularisation_parameter, alpha1, alpha0, iterations, LipshitzConst)
    elif inputData.ndim == 3:
        return 0
# Directional Total-variation Fast-Gradient-Projection (FGP)
def dTV_FGP_GPU(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg,
                     printM):
    if inputData.ndim == 2:
        return FGPdTV2D(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg,
                     printM)
    elif inputData.ndim == 3:
        return FGPdTV3D(inputData,
                     refdata,
                     regularisation_parameter,
                     iterations, 
                     tolerance_param,
                     eta_const,
                     methodTV,
                     nonneg,
                     printM)
# Nonlocal Isotropic Diffusion (NDF)
def NDF_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter,
                     penalty_type):
    if inputData.ndim == 2:
        return NDF_GPU_2D(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter,
                     penalty_type)
    elif inputData.ndim == 3:
        return NDF_GPU_3D(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter,
                     penalty_type)
# Anisotropic Fourth-Order diffusion
def Diff4th_GPU(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter):
    if inputData.ndim == 2:
        return Diff4th_2D(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter)
    elif inputData.ndim == 3:
        return Diff4th_3D(inputData,
                     regularisation_parameter,
                     edge_parameter,
                     iterations, 
                     time_marching_parameter)
                     
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
                       dims[1], dims[0], 1);   
     
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
                       dims[2], dims[1], dims[0]);   
     
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
                       dims[1], dims[0], 1);   
     
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
    TV_FGP_GPU_main(&inputData[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0]);   
     
    return outputData 
#***************************************************************#
#********************** Total-variation SB *********************#
#***************************************************************#
#*************** Total-variation Split Bregman (SB)*************#
def SBTV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     int methodTV,
                     int printM):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    TV_SB_GPU_main(&inputData[0,0], &outputData[0,0],                        
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[1], dims[0], 1);   
     
    return outputData
    
def SBTV3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     int methodTV,
                     int printM):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
          
    # Running CUDA code here    
    TV_SB_GPU_main(&inputData[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[2], dims[1], dims[0]);
     
    return outputData 

#***************************************************************#
#***************** Total Generalised Variation *****************#
#***************************************************************#
def TGV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     float alpha1,
                     float alpha0,
                     int iterationsNumb, 
                     float LipshitzConst):
                         
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    #/* Run TGV iterations for 2D data */
    TGV_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, 
                       alpha1,
                       alpha0,
                       iterationsNumb, 
                       LipshitzConst,
                       dims[1],dims[0])
    return outputData

#****************************************************************#
#**************Directional Total-variation FGP ******************#
#****************************************************************#
#******** Directional TV Fast-Gradient-Projection (FGP)*********#
def FGPdTV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
             np.ndarray[np.float32_t, ndim=2, mode="c"] refdata,
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     float eta_const,
                     int methodTV,
                     int nonneg,
                     int printM):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    dTV_FGP_GPU_main(&inputData[0,0], &refdata[0,0], &outputData[0,0],                        
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       eta_const,
                       methodTV,
                       nonneg,
                       printM,
                       dims[1], dims[0], 1);   
     
    return outputData
    
def FGPdTV3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
             np.ndarray[np.float32_t, ndim=3, mode="c"] refdata, 
                     float regularisation_parameter,
                     int iterations, 
                     float tolerance_param,
                     float eta_const,
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
    dTV_FGP_GPU_main(&inputData[0,0,0], &refdata[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       eta_const,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0]);
    return outputData 

#****************************************************************#
#***************Nonlinear (Isotropic) Diffusion******************#
#****************************************************************#
def NDF_GPU_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,                     
                     float time_marching_parameter,
                     int penalty_type):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
    
    #rangecheck = penalty_type < 1 and penalty_type > 3
    #if not rangecheck:
#        raise ValueError('Choose penalty type as 1 for Huber, 2 - Perona-Malik, 3 - Tukey Biweight')
    
    # Run Nonlinear Diffusion iterations for 2D data 
    # Running CUDA code here  
    NonlDiff_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)
    return outputData
            
def NDF_GPU_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,                     
                     float time_marching_parameter,
                     int penalty_type):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')    
       
    # Run Nonlinear Diffusion iterations for  3D data 
    # Running CUDA code here  
    NonlDiff_GPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])

    return outputData
#****************************************************************#
#************Anisotropic Fourth-Order diffusion******************#
#****************************************************************#
def Diff4th_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,
                     float time_marching_parameter):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
    
    # Run Anisotropic Fourth-Order diffusion for 2D data 
    # Running CUDA code here  
    Diffus4th_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[1], dims[0], 1)
    return outputData
            
def Diff4th_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,
                     float time_marching_parameter):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')    
       
    # Run Anisotropic Fourth-Order diffusion for  3D data 
    # Running CUDA code here  
    Diffus4th_GPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[2], dims[1], dims[0])

    return outputData

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

CUDAErrorMessage = 'CUDA error'

cdef extern int TV_ROF_GPU_main(float* Input, float* Output, float lambdaPar, int iter, float tau, int N, int M, int Z);
cdef extern int TV_FGP_GPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int nonneg, int printM, int N, int M, int Z);
cdef extern int TV_SB_GPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int printM, int N, int M, int Z);
cdef extern int TGV_GPU_main(float *Input, float *Output, float lambdaPar, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY);
cdef extern int LLT_ROF_GPU_main(float *Input, float *Output, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau, int N, int M, int Z);
cdef extern int NonlDiff_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int N, int M, int Z);
cdef extern int dTV_FGP_GPU_main(float *Input, float *InputRef, float *Output, float lambdaPar, int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int printM, int N, int M, int Z);
cdef extern int Diffus4th_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int N, int M, int Z);
cdef extern int PatchSelect_GPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, float *Weights, int N, int M, int SearchWindow, int SimilarWin, int NumNeighb, float h);

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
# LLT-ROF model
def LLT_ROF_GPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter):
    if inputData.ndim == 2:
        return LLT_ROF_GPU2D(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter)
    elif inputData.ndim == 3:
        return LLT_ROF_GPU3D(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter)
# Total Generilised Variation (TGV)
def TGV_GPU(inputData, regularisation_parameter, alpha1, alpha0, iterations, LipshitzConst):
    if inputData.ndim == 2:
        return TGV2D(inputData, regularisation_parameter, alpha1, alpha0, iterations, LipshitzConst)
    elif inputData.ndim == 3:
        shape = inputData.shape
        out = inputData.copy()
        for i in range(shape[0]):
            out[i,:,:] = TGV2D(inputData[i,:,:], regularisation_parameter, 
               alpha1, alpha0, iterations, LipshitzConst)
        return out
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
    if (TV_ROF_GPU_main(
            &inputData[0,0], &outputData[0,0], 
                       regularisation_parameter,
                       iterations , 
                       time_marching_parameter, 
                       dims[1], dims[0], 1)==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);
    
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
    if (TV_ROF_GPU_main(
            &inputData[0,0,0], &outputData[0,0,0], 
                       regularisation_parameter,
                       iterations , 
                       time_marching_parameter, 
                       dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);
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
    if (TV_FGP_GPU_main(&inputData[0,0], &outputData[0,0],
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[1], dims[0], 1)==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

    
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
    if (TV_FGP_GPU_main(&inputData[0,0,0], &outputData[0,0,0],
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

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
    if (TV_SB_GPU_main(&inputData[0,0], &outputData[0,0],
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[1], dims[0], 1)==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

    
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
    if (TV_SB_GPU_main(&inputData[0,0,0], &outputData[0,0,0],
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);


#***************************************************************#
#************************ LLT-ROF model ************************#
#***************************************************************#
#************Joint LLT-ROF model for higher order **************#
def LLT_ROF_GPU2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameterROF,
                     float regularisation_parameterLLT,
                     int iterations, 
                     float time_marching_parameter):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    if (LLT_ROF_GPU_main(&inputData[0,0], &outputData[0,0],regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, dims[1],dims[0],1)==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

    
def LLT_ROF_GPU3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameterROF,
                     float regularisation_parameterLLT,
                     int iterations, 
                     float time_marching_parameter):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
          
    # Running CUDA code here    
    if (LLT_ROF_GPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);



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
    if (TGV_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter,
                       alpha1,
                       alpha0,
                       iterationsNumb, 
                       LipshitzConst,
                       dims[1],dims[0])==0):
        return outputData
    else:
        raise ValueError(CUDAErrorMessage);


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
    if (dTV_FGP_GPU_main(&inputData[0,0], &refdata[0,0], &outputData[0,0],
                       regularisation_parameter, 
                       iterations, 
                       tolerance_param,
                       eta_const,
                       methodTV,
                       nonneg,
                       printM,
                       dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError(CUDAErrorMessage);

    
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
    if (dTV_FGP_GPU_main(&inputData[0,0,0], &refdata[0,0,0], &outputData[0,0,0],
                       regularisation_parameter , 
                       iterations, 
                       tolerance_param,
                       eta_const,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);


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
    if (NonlDiff_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

            
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
    if (NonlDiff_GPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

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
    if (Diffus4th_GPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError(CUDAErrorMessage);

            
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
    if (Diffus4th_GPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[2], dims[1], dims[0])==0):
        return outputData;
    else:
        raise ValueError(CUDAErrorMessage);

#****************************************************************#
#************Patch-based weights pre-selection******************#
#****************************************************************#
def PATCHSEL_GPU(inputData, searchwindow, patchwindow, neighbours, edge_parameter):
    if inputData.ndim == 2:
        return PatchSel_2D(inputData, searchwindow, patchwindow, neighbours, edge_parameter)
    elif inputData.ndim == 3:
        return 1
def PatchSel_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
                     int searchwindow,
                     int patchwindow,
                     int neighbours,
                     float edge_parameter):
    cdef long dims[3]
    dims[0] = neighbours
    dims[1] = inputData.shape[0]
    dims[2] = inputData.shape[1]    
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Weights = \
            np.zeros([dims[0], dims[1],dims[2]], dtype='float32')
    
    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] H_i = \
            np.zeros([dims[0], dims[1],dims[2]], dtype='uint16')
            
    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] H_j = \
            np.zeros([dims[0], dims[1],dims[2]], dtype='uint16')

    # Run patch-based weight selection function
    if (PatchSelect_GPU_main(&inputData[0,0], &H_j[0,0,0], &H_i[0,0,0], &Weights[0,0,0], dims[2], dims[1], searchwindow, patchwindow,  neighbours,  edge_parameter)==0):
        return H_i, H_j, Weights;
    else:
        raise ValueError(CUDAErrorMessage);


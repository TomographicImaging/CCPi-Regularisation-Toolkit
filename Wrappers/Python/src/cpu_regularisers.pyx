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

cdef extern float TV_ROF_CPU_main(float *Input, float *Output, float lambdaPar, int iterationsNumb, float tau, int dimX, int dimY, int dimZ);
cdef extern float TV_FGP_CPU_main(float *Input, float *Output, float lambdaPar, int iterationsNumb, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);
cdef extern float SB_TV_CPU_main(float *Input, float *Output, float lambdaPar, int iterationsNumb, float epsil, int methodTV, int printM, int dimX, int dimY, int dimZ);
cdef extern float LLT_ROF_CPU_main(float *Input, float *Output, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau, int dimX, int dimY, int dimZ);
cdef extern float TGV_main(float *Input, float *Output, float lambdaPar, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY, int dimZ);
cdef extern float Diffusion_CPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int dimX, int dimY, int dimZ);
cdef extern float Diffus4th_CPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int dimX, int dimY, int dimZ);
cdef extern float TNV_CPU_main(float *Input, float *u, float lambdaPar, int maxIter, float tol, int dimX, int dimY, int dimZ);
cdef extern float dTV_FGP_CPU_main(float *Input, float *InputRef, float *Output, float lambdaPar, int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);
cdef extern float PatchSelect_CPU_main(float *Input, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, int dimX, int dimY, int dimZ, int SearchWindow, int SimilarWin, int NumNeighb, float h, int switchM);
cdef extern float Nonlocal_TV_CPU_main(float *A_orig, float *Output, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, int dimX, int dimY, int dimZ, int NumNeighb, float lambdaReg, int IterNumb);

cdef extern float Diffusion_Inpaint_CPU_main(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int dimX, int dimY, int dimZ);
cdef extern float NonlocalMarching_Inpaint_main(float *Input, unsigned char *M, float *Output, unsigned char *M_upd, int SW_increment, int iterationsNumb, int trigger, int dimX, int dimY, int dimZ);
cdef extern float TV_energy2D(float *U, float *U0, float *E_val, float lambdaPar, int type, int dimX, int dimY);
cdef extern float TV_energy3D(float *U, float *U0, float *E_val, float lambdaPar, int type, int dimX, int dimY, int dimZ);
#****************************************************************#
#********************** Total-variation ROF *********************#
#****************************************************************#
def TV_ROF_CPU(inputData, regularisation_parameter, iterationsNumb, marching_step_parameter):
    if inputData.ndim == 2:
        return TV_ROF_2D(inputData, regularisation_parameter, iterationsNumb, marching_step_parameter)
    elif inputData.ndim == 3:
        return TV_ROF_3D(inputData, regularisation_parameter, iterationsNumb, marching_step_parameter)

def TV_ROF_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb,                     
                     float marching_step_parameter):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    # Run ROF iterations for 2D data 
    TV_ROF_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, iterationsNumb, marching_step_parameter, dims[1], dims[0], 1)
    
    return outputData
            
def TV_ROF_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb,
                     float marching_step_parameter):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
           
    # Run ROF iterations for 3D data 
    TV_ROF_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, iterationsNumb, marching_step_parameter, dims[2], dims[1], dims[0])

    return outputData

#****************************************************************#
#********************** Total-variation FGP *********************#
#****************************************************************#
#******** Total-variation Fast-Gradient-Projection (FGP)*********#
def TV_FGP_CPU(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, nonneg, printM):
    if inputData.ndim == 2:
        return TV_FGP_2D(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, nonneg, printM)
    elif inputData.ndim == 3:
        return TV_FGP_3D(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, nonneg, printM)

def TV_FGP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb, 
                     float tolerance_param,
                     int methodTV,
                     int nonneg,
                     int printM):
                         
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    #/* Run FGP-TV iterations for 2D data */
    TV_FGP_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, 
                       iterationsNumb, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[1],dims[0],1)
    
    return outputData        
            
def TV_FGP_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb, 
                     float tolerance_param,
                     int methodTV,
                     int nonneg,
                     int printM):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0], dims[1], dims[2]], dtype='float32')
           
    #/* Run FGP-TV iterations for 3D data */
    TV_FGP_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter,
                       iterationsNumb, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0])
    return outputData 

#***************************************************************#
#********************** Total-variation SB *********************#
#***************************************************************#
#*************** Total-variation Split Bregman (SB)*************#
def TV_SB_CPU(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, printM):
    if inputData.ndim == 2:
        return TV_SB_2D(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, printM)
    elif inputData.ndim == 3:
        return TV_SB_3D(inputData, regularisation_parameter, iterationsNumb, tolerance_param, methodTV, printM)

def TV_SB_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb, 
                     float tolerance_param,
                     int methodTV,
                     int printM):
                         
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    #/* Run SB-TV iterations for 2D data */
    SB_TV_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, 
                       iterationsNumb, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[1],dims[0],1)
    
    return outputData        
            
def TV_SB_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb, 
                     float tolerance_param,
                     int methodTV,
                     int printM):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0], dims[1], dims[2]], dtype='float32')
           
    #/* Run SB-TV iterations for 3D data */
    SB_TV_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter,
                       iterationsNumb, 
                       tolerance_param,
                       methodTV,
                       printM,
                       dims[2], dims[1], dims[0])
    return outputData 

#***************************************************************#
#***************** Total Generalised Variation *****************#
#***************************************************************#
def TGV_CPU(inputData, regularisation_parameter, alpha1, alpha0, iterations, LipshitzConst):
    if inputData.ndim == 2:
        return TGV_2D(inputData, regularisation_parameter, alpha1, alpha0, 
                      iterations, LipshitzConst)
    elif inputData.ndim == 3:
        return TGV_3D(inputData, regularisation_parameter, alpha1, alpha0, 
                      iterations, LipshitzConst)

def TGV_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
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
    TGV_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, 
                       alpha1,
                       alpha0,
                       iterationsNumb, 
                       LipshitzConst,
                       dims[1],dims[0],1)
    return outputData
def TGV_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     float alpha1,
                     float alpha0,
                     int iterationsNumb, 
                     float LipshitzConst):
                         
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0], dims[1], dims[2]], dtype='float32')
                   
    #/* Run TGV iterations for 3D data */
    TGV_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, 
                       alpha1,
                       alpha0,
                       iterationsNumb, 
                       LipshitzConst,
                       dims[2], dims[1], dims[0])
    return outputData

#***************************************************************#
#******************* ROF - LLT regularisation ******************#
#***************************************************************#
def LLT_ROF_CPU(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter):
    if inputData.ndim == 2:
        return LLT_ROF_2D(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter)
    elif inputData.ndim == 3:
        return LLT_ROF_3D(inputData, regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter)

def LLT_ROF_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularisation_parameterROF,
                     float regularisation_parameterLLT,
                     int iterations, 
                     float time_marching_parameter):
                         
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    #/* Run ROF-LLT iterations for 2D data */
    LLT_ROF_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, dims[1],dims[0],1)
    return outputData

def LLT_ROF_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameterROF,
                     float regularisation_parameterLLT,
                     int iterations, 
                     float time_marching_parameter):
						 
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0], dims[1], dims[2]], dtype='float32')
           
    #/* Run ROF-LLT iterations for 3D data */
    LLT_ROF_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameterROF, regularisation_parameterLLT, iterations, time_marching_parameter, dims[2], dims[1], dims[0])
    return outputData 

#****************************************************************#
#**************Directional Total-variation FGP ******************#
#****************************************************************#
#******** Directional TV Fast-Gradient-Projection (FGP)*********#
def dTV_FGP_CPU(inputData, refdata, regularisation_parameter, iterationsNumb, tolerance_param, eta_const, methodTV, nonneg, printM):
    if inputData.ndim == 2:
        return dTV_FGP_2D(inputData, refdata, regularisation_parameter, iterationsNumb, tolerance_param, eta_const, methodTV, nonneg, printM)
    elif inputData.ndim == 3:
        return dTV_FGP_3D(inputData, refdata, regularisation_parameter, iterationsNumb, tolerance_param, eta_const, methodTV, nonneg, printM)

def dTV_FGP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
               np.ndarray[np.float32_t, ndim=2, mode="c"] refdata,
                     float regularisation_parameter,
                     int iterationsNumb, 
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
                   
    #/* Run FGP-dTV iterations for 2D data */
    dTV_FGP_CPU_main(&inputData[0,0], &refdata[0,0], &outputData[0,0], regularisation_parameter, 
                       iterationsNumb, 
                       tolerance_param,
                       eta_const,
                       methodTV,                       
                       nonneg,
                       printM,
                       dims[1], dims[0], 1)
    
    return outputData        
            
def dTV_FGP_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
               np.ndarray[np.float32_t, ndim=3, mode="c"] refdata,
                     float regularisation_parameter,
                     int iterationsNumb, 
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
            np.zeros([dims[0], dims[1], dims[2]], dtype='float32')
           
    #/* Run FGP-dTV iterations for 3D data */
    dTV_FGP_CPU_main(&inputData[0,0,0], &refdata[0,0,0], &outputData[0,0,0], regularisation_parameter,
                       iterationsNumb, 
                       tolerance_param,
                       eta_const,
                       methodTV,
                       nonneg,
                       printM,
                       dims[2], dims[1], dims[0])
    return outputData
    
#****************************************************************#
#*********************Total Nuclear Variation********************#
#****************************************************************#
def TNV_CPU(inputData, regularisation_parameter, iterationsNumb, tolerance_param):
    if inputData.ndim == 2:
        return 
    elif inputData.ndim == 3:
        return TNV_3D(inputData, regularisation_parameter, iterationsNumb, tolerance_param)

def TNV_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularisation_parameter,
                     int iterationsNumb,
                     float tolerance_param):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
           
    # Run TNV iterations for 3D (X,Y,Channels) data 
    TNV_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, iterationsNumb, tolerance_param, dims[2], dims[1], dims[0])
    return outputData
#****************************************************************#
#***************Nonlinear (Isotropic) Diffusion******************#
#****************************************************************#
def NDF_CPU(inputData, regularisation_parameter, edge_parameter, iterationsNumb,time_marching_parameter, penalty_type):
    if inputData.ndim == 2:
        return NDF_2D(inputData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)
    elif inputData.ndim == 3:
        return NDF_3D(inputData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)

def NDF_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
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
    
    # Run Nonlinear Diffusion iterations for 2D data 
    Diffusion_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)
    return outputData
            
def NDF_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
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
    Diffusion_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])

    return outputData

#****************************************************************#
#*************Anisotropic Fourth-Order diffusion*****************#
#****************************************************************#
def Diff4th_CPU(inputData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter):
    if inputData.ndim == 2:
        return Diff4th_2D(inputData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter)
    elif inputData.ndim == 3:
        return Diff4th_3D(inputData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter)

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
    Diffus4th_CPU_main(&inputData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[1], dims[0], 1)
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
    Diffus4th_CPU_main(&inputData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, dims[2], dims[1], dims[0])

    return outputData

#****************************************************************#
#***************Patch-based weights calculation******************#
#****************************************************************#
def PATCHSEL_CPU(inputData, searchwindow, patchwindow, neighbours, edge_parameter):
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
    PatchSelect_CPU_main(&inputData[0,0], &H_j[0,0,0], &H_i[0,0,0], &H_i[0,0,0], &Weights[0,0,0], dims[2], dims[1], 0, searchwindow, patchwindow,  neighbours,  edge_parameter, 1)
    return H_i, H_j, Weights
"""
def PatchSel_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
                     int searchwindow,
                     int patchwindow,
                     int neighbours,
                     float edge_parameter):
    cdef long dims[4]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    dims[3] = neighbours
    
    cdef np.ndarray[np.float32_t, ndim=4, mode="c"] Weights = \
            np.zeros([dims[3],dims[0],dims[1],dims[2]], dtype='float32')
    
    cdef np.ndarray[np.uint16_t, ndim=4, mode="c"] H_i = \
            np.zeros([dims[3],dims[0],dims[1],dims[2]], dtype='uint16')
            
    cdef np.ndarray[np.uint16_t, ndim=4, mode="c"] H_j = \
            np.zeros([dims[3],dims[0],dims[1],dims[2]], dtype='uint16')
            
    cdef np.ndarray[np.uint16_t, ndim=4, mode="c"] H_k = \
            np.zeros([dims[3],dims[0],dims[1],dims[2]], dtype='uint16')

    # Run patch-based weight selection function
    PatchSelect_CPU_main(&inputData[0,0,0], &H_i[0,0,0,0], &H_j[0,0,0,0], &H_k[0,0,0,0], &Weights[0,0,0,0], dims[2], dims[1], dims[0], searchwindow, patchwindow,  neighbours, edge_parameter, 1)
    return H_i, H_j, H_k, Weights
"""

#****************************************************************#
#***************Non-local Total Variation******************#
#****************************************************************#
def NLTV_CPU(inputData, H_i, H_j, H_k, Weights, regularisation_parameter, iterations):
    if inputData.ndim == 2:
        return NLTV_2D(inputData, H_i, H_j, Weights, regularisation_parameter, iterations)
    elif inputData.ndim == 3:
        return 1
def NLTV_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
                     np.ndarray[np.uint16_t, ndim=3, mode="c"] H_i,
                     np.ndarray[np.uint16_t, ndim=3, mode="c"] H_j,
                     np.ndarray[np.float32_t, ndim=3, mode="c"] Weights,
                     float regularisation_parameter,
                     int iterations):

    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    neighbours = H_i.shape[0]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')
    
    # Run nonlocal TV regularisation
    Nonlocal_TV_CPU_main(&inputData[0,0], &outputData[0,0], &H_i[0,0,0], &H_j[0,0,0], &H_i[0,0,0], &Weights[0,0,0], dims[1], dims[0], 0, neighbours, regularisation_parameter, iterations)
    return outputData

#*********************Inpainting WITH****************************#
#***************Nonlinear (Isotropic) Diffusion******************#
#****************************************************************#
def NDF_INPAINT_CPU(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type):
    if inputData.ndim == 2:
        return NDF_INP_2D(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)
    elif inputData.ndim == 3:
        return NDF_INP_3D(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)

def NDF_INP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
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
    
    # Run Inpaiting by Diffusion iterations for 2D data 
    Diffusion_Inpaint_CPU_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)
    return outputData
            
def NDF_INP_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
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
    
    # Run Inpaiting by Diffusion iterations for 3D data 
    Diffusion_Inpaint_CPU_main(&inputData[0,0,0], &maskData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])

    return outputData
#*********************Inpainting WITH****************************#
#***************Nonlocal Vertical Marching method****************#
#****************************************************************#
def NVM_INPAINT_CPU(inputData, maskData, SW_increment, iterationsNumb):
    if inputData.ndim == 2:
        return NVM_INP_2D(inputData, maskData, SW_increment, iterationsNumb)
    elif inputData.ndim == 3:
        return 

def NVM_INP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
               np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                     int SW_increment,
                     int iterationsNumb):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')   
    
    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')
    
    # Run Inpaiting by Nonlocal vertical marching method for 2D data 
    NonlocalMarching_Inpaint_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], 
                                  &maskData_upd[0,0],
                                  SW_increment, iterationsNumb, 1, dims[1], dims[0], 1)
    
    return (outputData, maskData_upd)


#****************************************************************#
#***************Calculation of TV-energy functional**************#
#****************************************************************#
def TV_ENERGY(inputData, inputData0, regularisation_parameter, typeFunctional):
    if inputData.ndim == 2:
        return TV_ENERGY_2D(inputData, inputData0, regularisation_parameter, typeFunctional)
    elif inputData.ndim == 3:
        return TV_ENERGY_3D(inputData, inputData0, regularisation_parameter, typeFunctional)

def TV_ENERGY_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                 np.ndarray[np.float32_t, ndim=2, mode="c"] inputData0, 
                     float regularisation_parameter,
                     int typeFunctional):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] outputData = \
            np.zeros([1], dtype='float32')
                   
    # run function    
    TV_energy2D(&inputData[0,0], &inputData0[0,0], &outputData[0], regularisation_parameter, typeFunctional, dims[1], dims[0])
    
    return outputData
            
def TV_ENERGY_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
                 np.ndarray[np.float32_t, ndim=3, mode="c"] inputData0, 
                     float regularisation_parameter,
                     int typeFunctional):
						 
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] outputData = \
            np.zeros([1], dtype='float32')
           
    # Run function
    TV_energy3D(&inputData[0,0,0], &inputData0[0,0,0], &outputData[0], regularisation_parameter, typeFunctional, dims[2], dims[1], dims[0])

    return outputData

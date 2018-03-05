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

Author: Edoardo Pasca
"""

import cython
import numpy as np
cimport numpy as np

cdef extern float TV_ROF_CPU_main(float *Input, float *Output, int dimX, int dimY, int dimZ, int iterationsNumb, float tau, float flambda);
cdef extern float TV_FGP_CPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);								  

# Can we use the same name here in "def" as the C function? 
def TV_ROF_CPU(inputData, iterations, regularization_parameter, 
           marching_step_parameter):
    if inputData.ndim == 2:
        return TV_ROF_2D(inputData, iterations, regularization_parameter, 
                         marching_step_parameter)
    elif inputData.ndim == 3:
        return TV_ROF_3D(inputData, iterations, regularization_parameter, 
                         marching_step_parameter)

def TV_ROF_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     int iterations,
                     float regularization_parameter,
                     float marching_step_parameter
                     ):
                         
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B = \
            np.zeros([dims[0],dims[1]], dtype='float32')
                   
    #/* Run ROF iterations for 2D data */
    TV_ROF_CPU_main(&inputData[0,0], &B[0,0], dims[0], dims[1], 1, iterations,
               marching_step_parameter, regularization_parameter)
    
    return B        
            
def TV_ROF_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     int iterations,
                     float regularization_parameter,
                     float marching_step_parameter
                     ):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
           
    #/* Run ROF iterations for 3D data */
    TV_ROF_CPU_main(&inputData[0,0,0], &B[0,0,0], dims[0], dims[1], dims[2], iterations,
             marching_step_parameter, regularization_parameter)

    return B    
                     
def TV_FGP_CPU(inputData, regularization_parameter, iterations, tolerance_param, methodTV, nonneg, printM):
    if inputData.ndim == 2:
        return TV_FGP_2D(inputData, regularization_parameter, iterations, tolerance_param, methodTV, nonneg, printM)
    elif inputData.ndim == 3:
        return TV_FGP_3D(inputData, regularization_parameter, iterations, tolerance_param, methodTV, nonneg, printM)

def TV_FGP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularization_parameter,
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
                   
    #/* Run ROF iterations for 2D data */
    TV_FGP_CPU_main(&inputData[0,0], &outputData[0,0], regularization_parameter, 
                       iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[0], dims[1], 1)
    
    return outputData        
            
def TV_FGP_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
 float regularization_parameter,
                     int iterations, 
                     float tolerance_param,
                     int methodTV,
                     int nonneg,
                     int printM):
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
           
    #/* Run ROF iterations for 3D data */
    TV_FGP_CPU_main(&inputData[0,0,0], &outputData[0,0, 0], regularization_parameter,
					   iterations, 
                       tolerance_param,
                       methodTV,
                       nonneg,
                       printM,
                       dims[0], dims[1], dims[2])

    return outputData 

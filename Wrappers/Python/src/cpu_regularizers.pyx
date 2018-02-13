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

cdef extern float TV_main(float *D1, float *D2, float *D3, float *B, float *A, 
       float lambda, float tau, int dimY, int dimX, int dimZ);
cdef extern float D1_func(float *A, float *D1, int dimY, int dimX, int dimZ);
cdef extern float D2_func(float *A, float *D2, int dimY, int dimX, int dimZ);
cdef extern float D3_func(float *A, float *D3, int dimY, int dimX, int dimZ);
cdef extern void copyIm (float *A, float *U, int dimX, int dimY, int dimZ);


def ROF_TV(inputData, iterations, regularization_parameter, marching_step_parameter):
    if inputData.ndim == 2:
	    return ROF_TV_2D(inputData, iterations, regularization_parameter, 
		                 marching_step_parameter)
	elif inputData.ndim == 3:
	    return ROF_TV_3D(inputData, iterations, regularization_parameter, 
		                 marching_step_parameter)

def ROF_TV_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     int iterations,
					 float regularization_parameter
					 float marching_step_parameter
                     ):
	cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
	
	cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
	cdef np.ndarray[np.float32_t, ndim=2, mode="c"] D1 = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
	cdef np.ndarray[np.float32_t, ndim=2, mode="c"] D2 = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
			
	copyIm(&inputData[0,0], &B[0,0], dims[0], dims[1], 1);
	#/* start TV iterations */
	cdef int i = 0;
    for i in range(iterations): 
            
        #/* calculate differences */
        D1_func(&B[0,0], &D1[0,0], dims[0], dims[1], 1);
        D2_func(&B[0,0], &D2[0,0], dims[0], dims[1], 1);
            
        #/* calculate divergence and image update*/
        TV_main(&D1[0,0], &D2[0,0], &D2[0,0], &B[0,0], &A[0,0], 
		        regularization_parameter, marching_step_parameter, 
				dims[0], dims[1], 1)
	return B
        
			
def ROF_TV_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     int iterations,
					 float regularization_parameter
					 float marching_step_parameter
                     ):
	cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
	dims[2] = inputData.shape[2]
	
	cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
	cdef np.ndarray[np.float32_t, ndim=3, mode="c"] D1 = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
	cdef np.ndarray[np.float32_t, ndim=3, mode="c"] D2 = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
	cdef np.ndarray[np.float32_t, ndim=3, mode="c"] D3 = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
			
	copyIm(&inputData[0,0,0], &B[0,0,0], dims[0], dims[1], dims[2]);
	#/* start TV iterations */
	cdef int i = 0;
    for i in range(iterations): 
            
        #/* calculate differences */
        D1_func(&B[0,0,0], &D1[0,0,0], dims[0], dims[1], dims[2]);
        D2_func(&B[0,0,0], &D2[0,0,0], dims[0], dims[1], dims[2]);
		D3_func(&B[0,0,0], &D3[0,0,0], dims[0], dims[1], dims[2]);
            
        #/* calculate divergence and image update*/
        TV_main(&D1[0,0,0], &D2[0,0,0], &D3[0,0,0], &B[0,0,0], &A[0,0,0], 
		        regularization_parameter, marching_step_parameter, 
				dims[0], dims[1], dims[2])
	return B	
					 
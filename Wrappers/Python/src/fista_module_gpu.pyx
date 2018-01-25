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


cdef extern void Diff4th_GPU_kernel(float* A, float* B, int N, int M, int Z,
                            float sigma, int iter, float tau, float lambdaf);
cdef extern void NLM_GPU_kernel(float *A, float* B, float *Eucl_Vec, 
                                int N, int M,  int Z, int dimension, 
                                int SearchW, int SimilW, 
                                int SearchW_real, float denh2, float lambdaf);

def Diff4thHajiaboli(inputData, 
                     regularization_parameter, 
                     iterations, 
                     edge_preserving_parameter):
    if inputData.ndims == 2:
        return Diff4thHajiaboli2D(inputData,  
                     regularization_parameter, 
                     iterations, 
                     edge_preserving_parameter)
    elif inputData.ndims == 3:
        return Diff4thHajiaboli3D(inputData,  
                     regularization_parameter, 
                     iterations, 
                     edge_preserving_parameter)
                        
def Diff4thHajiaboli2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float regularization_parameter, 
                     int iterations, 
                     float edge_preserving_parameter):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    N = dims[0] + 2;
    M = dims[1] + 2;
    
    #time step is sufficiently small for an explicit methods
    tau = 0.01
    
    #A_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    #B_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    A_L = np.zeros((N,M), dtype=np.float)
    B_L = np.zeros((N,M), dtype=np.float)
    B = np.zeros((dims[0],dims[1]), dtype=np.float)
    #A = inputData

    # copy A to the bigger A_L with boundaries
    #pragma omp parallel for shared(A_L, A) private(i,j)
    cdef int i, j;
    for i in range(N):
        for j in range(M):
            if (((i > 0) and (i < N-1)) and  ((j > 0) and (j < M-1))):
                #A_L[i*M+j] = A[(i-1)*(dims[1])+(j-1)]
                A_L[i][j] = inputData[i-1][j-1]
        
    # Running CUDA code here
    #Diff4th_GPU_kernel(A_L, B_L, N, M, Z, (float)sigma, iter, (float)tau, lambda);    
#    Diff4th_GPU_kernel(
#            #<float*> A_L.data, <float*> B_L.data,
#            &A_L[0,0], &B_L[0,0], 
#                       N, M, 0, 
#                       edge_preserving_parameter,
#                       iterations , 
#                       tau, 
#                       regularization_parameter)
    # copy the processed B_L to a smaller B
    for i in range(N):
        for j in range(M):
            if (((i > 0) and (i < N-1)) and ((j > 0) and (j < M-1))):
                B[i-1][j-1] = B_L[i][j]
    ##pragma omp parallel for shared(B_L, B) private(i,j)
    #for (i=0; i < N; i++) {
    #    for (j=0; j < M; j++) {
    #        if (((i > 0) && (i < N-1)) &&  ((j > 0) && (j < M-1)))   B[(i-1)*(dims[1])+(j-1)] = B_L[i*M+j];
    #    }}
     
    return B

def Diff4thHajiaboli3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     float regularization_parameter, 
                     int iterations, 
                     float edge_preserving_parameter):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    N = dims[0] + 2
    M = dims[1] + 2
    Z = dims[2] + 2
    
    # Time Step is small for an explicit methods
    tau = 0.0007;
    
    #A_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    #B_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    A_L = np.zeros((N,M,Z), dtype=np.float)
    B_L = np.zeros((N,M,Z), dtype=np.float)
    B = np.zeros((dims[0],dims[1],dims[2]), dtype=np.float)
    #A = inputData
    
    # copy A to the bigger A_L with boundaries
    #pragma omp parallel for shared(A_L, A) private(i,j)
    cdef int i, j, k;
    for i in range(N):
        for j in range(M):
            for k in range(Z):
                if (((i > 0) and (i < N-1)) and \
                    ((j > 0) and (j < M-1)) and \
                    ((k > 0) and (k < Z-1))):
                        A_L[i][j][k] = inputData[i-1][j-1][k-1];
        
    # Running CUDA code here
    #Diff4th_GPU_kernel(A_L, B_L, N, M, Z, (float)sigma, iter, (float)tau, lambda);    
#    Diff4th_GPU_kernel(
#            #<float*> A_L.data, <float*> B_L.data,
#            &A_L[0,0,0], &B_L[0,0,0], 
#                       N, M, Z, 
#                       edge_preserving_parameter,
#                       iterations , 
#                       tau, 
#                       regularization_parameter)
    # copy the processed B_L to a smaller B
    for i in range(N):
        for j in range(M):
            for k in range(Z):
                if (((i > 0) and (i < N-1)) and \
                    ((j > 0) and (j < M-1)) and \
                    ((k > 0) and (k < Z-1))):
                    B[i-1][j-1][k-1] = B_L[i][j][k]
    
     
    return B

		

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
cdef extern void TV_ROF_GPU_kernel(float* A, float* B, int N, int M, int Z, int iter, float tau, float lambdaf);
cdef extern float pad_crop(float *A, float *Ap, 
                           int OldSizeX, int OldSizeY, int OldSizeZ, 
                           int NewSizeX, int NewSizeY, int NewSizeZ, 
                           int padXY, int switchpad_crop);

def Diff4thHajiaboli(inputData, 
                     edge_preserv_parameter, 
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter):
    if inputData.ndim == 2:
        return Diff4thHajiaboli2D(inputData,  
                     edge_preserv_parameter, 
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter)
    elif inputData.ndim == 3:
        return Diff4thHajiaboli3D(inputData,  
                     edge_preserv_parameter, 
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter)
        
def NML(inputData, 
                     SearchW_real, 
                     SimilW, 
                     h,
                     lambdaf):
    if inputData.ndim == 2:
        return NML2D(inputData, 
                     SearchW_real, 
                     SimilW, 
                     h,
                     lambdaf)
    elif inputData.ndim == 3:
        return NML3D(inputData, 
                     SearchW_real, 
                     SimilW, 
                     h,
                     lambdaf)

def GPU_ROF_TV(inputData,
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter):
    if inputData.ndim == 2:
        return ROFTV2D(inputData, 
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter)
    elif inputData.ndim == 3:
        return ROFTV3D(inputData, 
                     iterations, 
                     time_marching_parameter,
                     regularization_parameter)

                    
def Diff4thHajiaboli2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     float edge_preserv_parameter, 
                     int iterations, 
                     float time_marching_parameter,
                     float regularization_parameter):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    N = dims[0] + 2;
    M = dims[1] + 2;
    
    #A_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    #B_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] A_L = \
		    np.zeros([N,M], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B_L = \
		    np.zeros([N,M], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
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
    Diff4th_GPU_kernel(
            #<float*> A_L.data, <float*> B_L.data,
            &A_L[0,0], &B_L[0,0], 
                       N, M, 0, 
                       edge_preserv_parameter,
                       iterations , 
                       time_marching_parameter, 
                       regularization_parameter);
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
                     float edge_preserv_parameter, 
                     int iterations, 
                     float time_marching_parameter,
                     float regularization_parameter):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]
    N = dims[0] + 2
    M = dims[1] + 2
    Z = dims[2] + 2
    
    
    #A_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    #B_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] A_L = \
		    np.zeros([N,M,Z], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B_L = \
		    np.zeros([N,M,Z], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
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
    Diff4th_GPU_kernel(
            #<float*> A_L.data, <float*> B_L.data,
            &A_L[0,0,0], &B_L[0,0,0], 
                       N, M, Z, 
                       edge_preserv_parameter,
                       iterations , 
                       time_marching_parameter, 
                       regularization_parameter);
    # copy the processed B_L to a smaller B
    for i in range(N):
        for j in range(M):
            for k in range(Z):
                if (((i > 0) and (i < N-1)) and \
                    ((j > 0) and (j < M-1)) and \
                    ((k > 0) and (k < Z-1))):
                    B[i-1][j-1][k-1] = B_L[i][j][k]
    
     
    return B


def NML2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     SearchW_real, 
                     SimilW, 
                     h,
                     lambdaf):    
    N = inputData.shape[0]
    M = inputData.shape[1]      
    Z = 0
    numdims = inputData.ndim
     
    if h < 0:
        raise ValueError('Parameter for the PB filtering function must be > 0') 
             
    SearchW = SearchW_real + 2*SimilW;
    
    SearchW_full = 2*SearchW + 1; #/* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   #/* the full similarity window  size */
    h2 = h*h;
    
    padXY = SearchW + 2*SimilW; #/* padding sizes */
    newsizeX = N + 2*(padXY); #/* the X size of the padded array */
    newsizeY = M + 2*(padXY); #/* the Y size of the padded array */
    #newsizeZ = Z + 2*(padXY); #/* the Z size of the padded array */
    
    #output
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B = \
		    np.zeros([N,M], dtype='float32')
    #/*allocating memory for the padded arrays */
    
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Ap = \
		    np.zeros([newsizeX, newsizeY], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Bp = \
		    np.zeros([newsizeX, newsizeY], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] Eucl_Vec = \
		    np.zeros([SimilW_full*SimilW_full], dtype='float32')
    
    #/*Gaussian kernel */
    cdef int count, i_n, j_n;
    cdef float val;
    count = 0
    for i_n in range (-SimilW, SimilW +1):
        for j_n in range(-SimilW, SimilW +1):
            val = (float)(i_n*i_n + j_n*j_n)/(2*SimilW*SimilW)
            Eucl_Vec[count] = np.exp(-val)
            count = count + 1
    
    #/*Perform padding of image A to the size of [newsizeX * newsizeY] */
    switchpad_crop = 0; # /*padding*/
    pad_crop(&inputData[0,0], &Ap[0,0], M, N, 0, newsizeY, newsizeX, 0, padXY,
             switchpad_crop);
    
    #/* Do PB regularization with the padded array  */
    NLM_GPU_kernel(&Ap[0,0], &Bp[0,0], &Eucl_Vec[0], newsizeY, newsizeX, 0,
                   numdims, SearchW, SimilW, SearchW_real, 
                   h2, lambdaf);
    
    switchpad_crop = 1; #/*cropping*/
    pad_crop(&Bp[0,0], &B[0,0], M, N, 0, newsizeX, newsizeY, 0, padXY, 
             switchpad_crop)
    
    return B

def NML3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     SearchW_real, 
                     SimilW, 
                     h,
                     lambdaf):    
    N = inputData.shape[0]
    M = inputData.shape[1]      
    Z = inputData.shape[2]     
    numdims = inputData.ndim

    if h < 0:
        raise ValueError('Parameter for the PB filtering function must be > 0') 
             
    SearchW = SearchW_real + 2*SimilW;
    
    SearchW_full = 2*SearchW + 1; #/* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   #/* the full similarity window  size */
    h2 = h*h;
    
    padXY = SearchW + 2*SimilW; #/* padding sizes */
    newsizeX = N + 2*(padXY); #/* the X size of the padded array */
    newsizeY = M + 2*(padXY); #/* the Y size of the padded array */
    newsizeZ = Z + 2*(padXY); #/* the Z size of the padded array */
    
    #output
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B = \
		    np.zeros([N,M,Z], dtype='float32')
    #/*allocating memory for the padded arrays */
    
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Ap = \
		    np.zeros([newsizeX, newsizeY, newsizeZ], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Bp = \
		    np.zeros([newsizeX, newsizeY, newsizeZ], dtype='float32')
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] Eucl_Vec = \
		    np.zeros([SimilW_full*SimilW_full*SimilW_full],
				    dtype='float32')
    
    
    #/*Gaussian kernel */
    cdef int count, i_n, j_n, k_n;
    cdef float val;
    count = 0
    for i_n in range (-SimilW, SimilW +1):
        for j_n in range(-SimilW, SimilW +1):
            for k_n in range(-SimilW, SimilW+1):
                val = (i_n*i_n + j_n*j_n + k_n*k_n)/(2*SimilW*SimilW*SimilW)
                Eucl_Vec[count] = np.exp(-val)
                count = count + 1
    
    #/*Perform padding of image A to the size of [newsizeX * newsizeY] */
    switchpad_crop = 0; # /*padding*/
    pad_crop(&inputData[0,0,0], &Ap[0,0,0], 
             M, N, Z, 
             newsizeY, newsizeX, newsizeZ, 
             padXY,
             switchpad_crop);
    
    #/* Do PB regularization with the padded array  */
    NLM_GPU_kernel(&Ap[0,0,0], &Bp[0,0,0], &Eucl_Vec[0], 
                   newsizeY, newsizeX, newsizeZ,
                   numdims, SearchW, SimilW, SearchW_real, 
                   h2, lambdaf);
        
    switchpad_crop = 1; #/*cropping*/
    pad_crop(&Bp[0,0,0], &B[0,0,0],
             M, N, Z, 
             newsizeX, newsizeY, newsizeZ,
             padXY, 
             switchpad_crop)
    
    return B                   
                     
def ROFTV2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData, 
                     int iterations, 
                     float time_marching_parameter,
                     float regularization_parameter):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B = \
		    np.zeros([dims[0],dims[1]], dtype='float32')
          
    # Running CUDA code here    
    TV_ROF_GPU_kernel(            
            &inputData[0,0], &B[0,0], 
                       dims[0], dims[1], 1, 
                       iterations , 
                       time_marching_parameter, 
                       regularization_parameter);   
     
    return B
    
def ROFTV3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData, 
                     int iterations, 
                     float time_marching_parameter,
                     float regularization_parameter):
    
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] B = \
		    np.zeros([dims[0],dims[1],dims[2]], dtype='float32')
          
    # Running CUDA code here    
    TV_ROF_GPU_kernel(            
            &inputData[0,0,0], &B[0,0,0], 
                       dims[0], dims[1], dims[2], 
                       iterations , 
                       time_marching_parameter, 
                       regularization_parameter);   
     
    return B

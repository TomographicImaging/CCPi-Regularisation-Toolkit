#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include "Diff4th_GPU_kernel.h"

/*
 * 2D and 3D CUDA implementation of the 4th order PDE denoising model by Hajiaboli
 *
 * Reference :
 * "An anisotropic fourth-order diffusion filter for image noise removal" by M. Hajiaboli
 * 
 * Example
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .05*randn(size(Im)); % adding noise
 * u = Diff4thHajiaboli_GPU(single(u0), 0.02, 150);
 * subplot (1,2,1); imshow(u0,[ ]); title('Noisy Image')
 * subplot (1,2,2); imshow(u,[ ]); title('Denoised Image')
 *
 *
 * Linux/Matlab compilation:
 * compile in terminal: nvcc -Xcompiler -fPIC -shared -o Diff4th_GPU_kernel.o Diff4th_GPU_kernel.cu
 * then compile in Matlab: mex -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart Diff4thHajiaboli_GPU.cpp Diff4th_GPU_kernel.o
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int numdims, dimZ, size;
    float *A, *B, *A_L, *B_L;
    const int *dims;    
    
    numdims = mxGetNumberOfDimensions(prhs[0]);
    dims = mxGetDimensions(prhs[0]);       
    
    float sigma = (float)mxGetScalar(prhs[1]); /* edge-preserving parameter */
    float lambda = (float)mxGetScalar(prhs[2]); /* regularization parameter */
    int iter = (int)mxGetScalar(prhs[3]); /* iterations number */
    
    if (numdims == 2)  {
        
        int N, M, Z, i, j;
        Z = 0; // for the 2D case
        float tau = 0.01; // time step is sufficiently small for an explicit methods
        
        /*Input data*/
        A = (float*)mxGetData(prhs[0]);        
        N = dims[0] + 2;
        M = dims[1] + 2;
        A_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
        B_L = (float*)mxGetData(mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
        
        /*Output data*/
        B = (float*)mxGetData(plhs[0] = mxCreateNumericMatrix(dims[0], dims[1], mxSINGLE_CLASS, mxREAL));        
        
        // copy A to the bigger A_L with boundaries
        #pragma omp parallel for shared(A_L, A) private(i,j)
        for (i=0; i < N; i++) {
            for (j=0; j < M; j++) {
                if (((i > 0) && (i < N-1)) &&  ((j > 0) && (j < M-1)))  A_L[i*M+j] = A[(i-1)*(dims[1])+(j-1)];
            }}
        
        // Running CUDA code here
        Diff4th_GPU_kernel(A_L, B_L, N, M, Z, (float)sigma, iter, (float)tau, lambda);
        
        // copy the processed B_L to a smaller B
        #pragma omp parallel for shared(B_L, B) private(i,j)
        for (i=0; i < N; i++) {
            for (j=0; j < M; j++) {
                if (((i > 0) && (i < N-1)) &&  ((j > 0) && (j < M-1)))   B[(i-1)*(dims[1])+(j-1)] = B_L[i*M+j];
            }}
    }
    if (numdims == 3)  {
        //  3D image denoising / regularization
        int N, M, Z, i, j, k;
        float tau = 0.0007; // Time Step is small for an explicit methods
        A = (float*)mxGetData(prhs[0]);
        N = dims[0] + 2;
        M = dims[1] + 2;
        Z = dims[2] + 2;
        int N_dims[] = {N, M, Z};
        A_L = (float*)mxGetPr(mxCreateNumericArray(3, N_dims, mxSINGLE_CLASS, mxREAL));
        B_L = (float*)mxGetPr(mxCreateNumericArray(3, N_dims, mxSINGLE_CLASS, mxREAL));
        
        /* output data */
        B = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL));
        
        // copy A to the bigger A_L with boundaries
        #pragma omp parallel for shared(A_L, A) private(i,j,k)
        for (i=0; i < N; i++) {
            for (j=0; j < M; j++) {
                for (k=0; k < Z; k++) {
                    if (((i > 0) && (i < N-1)) &&  ((j > 0) && (j < M-1)) &&  ((k > 0) && (k < Z-1))) {
                        A_L[(N*M)*(k)+(i)*M+(j)] = A[(dims[0]*dims[1])*(k-1)+(i-1)*dims[1]+(j-1)];
                    }}}}
        
        // Running CUDA kernel here for diffusivity
        Diff4th_GPU_kernel(A_L, B_L, N, M, Z, (float)sigma, iter, (float)tau, lambda);
        
        // copy the processed B_L to a smaller B
        #pragma omp parallel for shared(B_L, B) private(i,j,k)
        for (i=0; i < N; i++) {
            for (j=0; j < M; j++) {
                for (k=0; k < Z; k++) {
                    if (((i > 0) && (i < N-1)) &&  ((j > 0) && (j < M-1)) &&  ((k > 0) && (k < Z-1))) {
                        B[(dims[0]*dims[1])*(k-1)+(i-1)*dims[1]+(j-1)] = B_L[(N*M)*(k)+(i)*M+(j)];
                    }}}}
    }
}
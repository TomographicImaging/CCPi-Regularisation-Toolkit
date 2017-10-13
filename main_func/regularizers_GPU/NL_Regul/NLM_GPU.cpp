#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include "NLM_GPU_kernel.h"

/* CUDA implementation of the patch-based (PB) regularization  for 2D and 3D images/volumes  
 * This method finds self-similar patches in data and performs one fixed point iteration to mimimize the PB penalty function
 * 
 * References: 1. Yang Z. & Jacob M. "Nonlocal Regularization of Inverse Problems"
 *             2. Kazantsev D. at. all "4D-CT reconstruction with unified spatial-temporal patch-based regularization"
 *
 * Input Parameters (mandatory):
 * 1. Image/volume (2D/3D)
 * 2. ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window)
 * 3. ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window)
 * 4. h - parameter for the PB penalty function
 * 5. lambda - regularization parameter 

 * Output:
 * 1. regularized (denoised) Image/volume (N x N x N)
 *
 * In matlab check what kind of GPU you have with "gpuDevice" command,
 * then set your ComputeCapability, here I use -arch compute_35
 *
 * Quick 2D denoising example in Matlab:   
   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
   ImDen = NLM_GPU(single(u0), 3, 2, 0.15, 1);
 
 * Linux/Matlab compilation:
 * compile in terminal: nvcc -Xcompiler -fPIC -shared -o NLM_GPU_kernel.o NLM_GPU_kernel.cu
 * then compile in Matlab: mex -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart NLM_GPU.cpp NLM_GPU_kernel.o
 *
 * D. Kazantsev 
 * 2014-17
 * Harwell/Manchester UK
 */

float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY,  int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop);

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int N, M, Z, i_n, j_n, k_n, numdims, SearchW, SimilW, SearchW_real, padXY, newsizeX, newsizeY, newsizeZ, switchpad_crop, count, SearchW_full, SimilW_full;
    const int  *dims;
    float *A, *B=NULL, *Ap=NULL, *Bp=NULL, *Eucl_Vec, h, h2, lambda, val, denh2;
    
    numdims = mxGetNumberOfDimensions(prhs[0]);
    dims = mxGetDimensions(prhs[0]);
    
    N = dims[0];
    M = dims[1];
    Z = dims[2];
    
    if ((numdims < 2) || (numdims > 3)) {mexErrMsgTxt("The input should be 2D image or 3D volume");}
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input in single precision is required"); }
    
    if(nrhs != 5) mexErrMsgTxt("Five inputs reqired: Image(2D,3D), SearchW, SimilW, Threshold, Regularization parameter");
    
    /*Handling inputs*/
    A  = (float *) mxGetData(prhs[0]);    /* the image to regularize/filter */
    SearchW_real  = (int) mxGetScalar(prhs[1]); /* the searching window ratio */
    SimilW =  (int) mxGetScalar(prhs[2]);  /* the similarity window ratio */
    h =  (float) mxGetScalar(prhs[3]);  /* parameter for the PB filtering function */
    lambda = (float) mxGetScalar(prhs[4]);
    
    if (h <= 0) mexErrMsgTxt("Parmeter for the PB penalty function should be > 0");
      
    SearchW = SearchW_real + 2*SimilW;
    
    SearchW_full = 2*SearchW + 1; /* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   /* the full similarity window  size */
    h2 = h*h;
    
    padXY = SearchW + 2*SimilW; /* padding sizes */
    newsizeX = N + 2*(padXY); /* the X size of the padded array */
    newsizeY = M + 2*(padXY); /* the Y size of the padded array */
    newsizeZ = Z + 2*(padXY); /* the Z size of the padded array */
    int N_dims[] = {newsizeX, newsizeY, newsizeZ};
    
    /******************************2D case ****************************/
    if (numdims == 2) {
        /*Handling output*/
        B = (float*)mxGetData(plhs[0] = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL));
        /*allocating memory for the padded arrays */
        Ap = (float*)mxGetData(mxCreateNumericMatrix(newsizeX, newsizeY, mxSINGLE_CLASS, mxREAL));
        Bp = (float*)mxGetData(mxCreateNumericMatrix(newsizeX, newsizeY, mxSINGLE_CLASS, mxREAL));
        Eucl_Vec = (float*)mxGetData(mxCreateNumericMatrix(SimilW_full*SimilW_full, 1, mxSINGLE_CLASS, mxREAL));
        
        /*Gaussian kernel */
        count = 0;
        for(i_n=-SimilW; i_n<=SimilW; i_n++) {
            for(j_n=-SimilW; j_n<=SimilW; j_n++) {
                val = (float)(i_n*i_n + j_n*j_n)/(2*SimilW*SimilW);
                Eucl_Vec[count] = exp(-val);
                count = count + 1;
            }} /*main neighb loop */
        
        /**************************************************************************/
        /*Perform padding of image A to the size of [newsizeX * newsizeY] */
        switchpad_crop = 0; /*padding*/
        pad_crop(A, Ap, M, N, 0, newsizeY, newsizeX, 0, padXY, switchpad_crop);
        
        /* Do PB regularization with the padded array  */
        NLM_GPU_kernel(Ap, Bp, Eucl_Vec, newsizeY, newsizeX, 0, numdims, SearchW, SimilW, SearchW_real, (float)h2, (float)lambda);
        
        switchpad_crop = 1; /*cropping*/
        pad_crop(Bp, B, M, N, 0, newsizeY, newsizeX, 0, padXY, switchpad_crop);
    }
    else
    {
        /******************************3D case ****************************/
        /*Handling output*/
        B = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL));
        /*allocating memory for the padded arrays */
        Ap = (float*)mxGetPr(mxCreateNumericArray(3, N_dims, mxSINGLE_CLASS, mxREAL));
        Bp = (float*)mxGetPr(mxCreateNumericArray(3, N_dims, mxSINGLE_CLASS, mxREAL));
        Eucl_Vec = (float*)mxGetData(mxCreateNumericMatrix(SimilW_full*SimilW_full*SimilW_full, 1, mxSINGLE_CLASS, mxREAL));
        
        /*Gaussian kernel */
        count = 0;
        for(i_n=-SimilW; i_n<=SimilW; i_n++) {
            for(j_n=-SimilW; j_n<=SimilW; j_n++) {
                for(k_n=-SimilW; k_n<=SimilW; k_n++) {
                    val = (float)(i_n*i_n + j_n*j_n + k_n*k_n)/(2*SimilW*SimilW*SimilW);
                    Eucl_Vec[count] = exp(-val);
                    count = count + 1;
                }}} /*main neighb loop */
        /**************************************************************************/
        /*Perform padding of image A to the size of [newsizeX * newsizeY * newsizeZ] */
        switchpad_crop = 0; /*padding*/
        pad_crop(A, Ap, M, N, Z, newsizeY, newsizeX, newsizeZ, padXY, switchpad_crop);
        
        /* Do PB regularization with the padded array  */
        NLM_GPU_kernel(Ap, Bp, Eucl_Vec, newsizeY, newsizeX, newsizeZ, numdims, SearchW, SimilW, SearchW_real, (float)h2, (float)lambda);
        
        switchpad_crop = 1; /*cropping*/
        pad_crop(Bp, B, M, N, Z, newsizeY, newsizeX, newsizeZ, padXY, switchpad_crop);
    } /*end else ndims*/
}

float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY, int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop)
{
    /* padding-cropping function */
    int i,j,k;    
    if (NewSizeZ > 1) {    
           for (i=0; i < NewSizeX; i++) {
            for (j=0; j < NewSizeY; j++) {
              for (k=0; k < NewSizeZ; k++) {
                if (((i >= padXY) && (i < NewSizeX-padXY)) &&  ((j >= padXY) && (j < NewSizeY-padXY)) &&  ((k >= padXY) && (k < NewSizeZ-padXY))) {
                    if (switchpad_crop == 0)  Ap[NewSizeX*NewSizeY*k + i*NewSizeY+j] = A[OldSizeX*OldSizeY*(k - padXY) + (i-padXY)*(OldSizeY)+(j-padXY)];
                    else  Ap[OldSizeX*OldSizeY*(k - padXY) + (i-padXY)*(OldSizeY)+(j-padXY)] = A[NewSizeX*NewSizeY*k + i*NewSizeY+j];
                }
            }}}   
    }
    else {
        for (i=0; i < NewSizeX; i++) {
            for (j=0; j < NewSizeY; j++) {
                if (((i >= padXY) && (i < NewSizeX-padXY)) &&  ((j >= padXY) && (j < NewSizeY-padXY))) {
                    if (switchpad_crop == 0)  Ap[i*NewSizeY+j] = A[(i-padXY)*(OldSizeY)+(j-padXY)];
                    else  Ap[(i-padXY)*(OldSizeY)+(j-padXY)] = A[i*NewSizeY+j];
                }
            }}
    }
    return *Ap;
}
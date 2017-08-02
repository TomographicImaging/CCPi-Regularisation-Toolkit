/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazanteev
Copyright 2017 Srikanth Nagella, Edoardo Pasca

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "mex.h"
#include "PatchBased_Regul_core.h"


/* C-OMP implementation of  patch-based (PB) regularization (2D and 3D cases). 
 * This method finds self-similar patches in data and performs one fixed point iteration to mimimize the PB penalty function
 * 
 * References: 1. Yang Z. & Jacob M. "Nonlocal Regularization of Inverse Problems"
 *             2. Kazantsev D. et al. "4D-CT reconstruction with unified spatial-temporal patch-based regularization"
 *
 * Input Parameters:
 * 1. Image (2D or 3D) [required]
 * 2. ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window) [optional]
 * 3. ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window) [optional]
 * 4. h - parameter for the PB penalty function [optional]
 * 5. lambda - regularization parameter  [optional]

 * Output:
 * 1. regularized (denoised) Image (N x N)/volume (N x N x N)
 *
 * 2D denoising example in Matlab:   
   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
   ImDen = PatchBased_Regul(single(u0), 3, 1, 0.08, 0.05); 
 *
 * Matlab + C/mex compilers needed
 * to compile with OMP support: mex PatchBased_Regul.c CFLAGS="\$CFLAGS -fopenmp -Wall" LDFLAGS="\$LDFLAGS -fopenmp"
 *
 * D. Kazantsev *
 * 02/07/2014
 * Harwell, UK
 */


void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) 
{    
    int N, M, Z, numdims, SearchW, SimilW, SearchW_real, padXY, newsizeX, newsizeY, newsizeZ, switchpad_crop;
    const int  *dims;
    float *A, *B=NULL, *Ap=NULL, *Bp=NULL, h, lambda;
    
    numdims = mxGetNumberOfDimensions(prhs[0]);
    dims = mxGetDimensions(prhs[0]);
    
    N = dims[0];
    M = dims[1];
    Z = dims[2];
    
    if ((numdims < 2) || (numdims > 3)) {mexErrMsgTxt("The input is 2D image or 3D volume");}
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input in single precision is required"); }
    
    if(nrhs != 5) mexErrMsgTxt("Five inputs reqired: Image(2D,3D), SearchW, SimilW, Threshold, Regularization parameter");
    
    /*Handling inputs*/
    A  = (float *) mxGetData(prhs[0]);    /* the image/volume to regularize/filter */
    SearchW_real = 3; /*default value*/
    SimilW = 1; /*default value*/
    h = 0.1; 
    lambda = 0.1;
    
    if ((nrhs == 2) || (nrhs == 3) || (nrhs == 4) || (nrhs == 5))   SearchW_real  = (int) mxGetScalar(prhs[1]); /* the searching window ratio */
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5))   SimilW =  (int) mxGetScalar(prhs[2]);  /* the similarity window ratio */
    if ((nrhs == 4) || (nrhs == 5))  h =  (float) mxGetScalar(prhs[3]);  /* parameter for the PB filtering function */
    if ((nrhs == 5))  lambda = (float) mxGetScalar(prhs[4]); /* regularization parameter */   


    if (h <= 0) mexErrMsgTxt("Parmeter for the PB penalty function should be > 0");
    if (lambda <= 0) mexErrMsgTxt(" Regularization parmeter should be > 0");
       
    SearchW = SearchW_real + 2*SimilW;
    
    /* SearchW_full = 2*SearchW + 1; */ /* the full searching window  size */
    /* SimilW_full = 2*SimilW + 1;  */  /* the full similarity window  size */
    
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
        /**************************************************************************/
        /*Perform padding of image A to the size of [newsizeX * newsizeY] */
        switchpad_crop = 0; /*padding*/
        pad_crop(A, Ap, M, N, 0, newsizeY, newsizeX, 0, padXY, switchpad_crop);
        
        /* Do PB regularization with the padded array  */
        PB_FUNC2D(Ap, Bp, newsizeY, newsizeX, padXY, SearchW, SimilW, (float)h, (float)lambda);
        
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
        /**************************************************************************/
        
        /*Perform padding of image A to the size of [newsizeX * newsizeY * newsizeZ] */
        switchpad_crop = 0; /*padding*/
        pad_crop(A, Ap, M, N, Z, newsizeY, newsizeX, newsizeZ, padXY, switchpad_crop);
        
        /* Do PB regularization with the padded array  */
        PB_FUNC3D(Ap, Bp, newsizeY, newsizeX, newsizeZ, padXY, SearchW, SimilW, (float)h, (float)lambda);
        
        switchpad_crop = 1; /*cropping*/
        pad_crop(Bp, B, M, N, Z, newsizeY, newsizeX, newsizeZ, padXY, switchpad_crop);
    } /*end else ndims*/ 
}    

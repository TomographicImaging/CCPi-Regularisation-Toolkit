/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "matrix.h"
#include "mex.h"
#include "ROF_TV_core.h"

/* C-OMP implementation of ROF-TV denoising/regularization model [1] (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. tau - marching step for explicit scheme, ~0.001 is recommended [REQUIRED]
 * 4. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
 *
 * Output:
 * [1] Regularized image/volume
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 * compile: mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * D. Kazantsev, 2016-18
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int i, number_of_dims, iter_numb, dimX, dimY, dimZ;
    const int  *dim_array;
    float *A, *B, *D1, *D2, *D3, lambda, tau;
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]);
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    tau =  (float) mxGetScalar(prhs[2]); /* marching step parameter */
    iter_numb =  (int) mxGetScalar(prhs[3]); /* iterations number */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 0; /*2D case*/
        B = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        /* copy into B */
        copyIm(A, B, dimX, dimY, 1);
        
        /* start TV iterations */
        for(i=0; i < iter_numb; i++) {
            
            /* calculate differences */
            D1_func(B, D1, dimX, dimY, dimZ);
            D2_func(B, D2, dimX, dimY, dimZ);
            
            /* calculate divergence and image update*/
            TV_main(D1, D2, D2, B, A, lambda, tau, dimX, dimY, dimZ);
        }
    }
    
    if (number_of_dims == 3) {
        B = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        
        /* copy into B */
        copyIm(A, B, dimX, dimY, dimZ);
        
        /* start TV iterations */
        for(i=0; i < iter_numb; i++) {
            
        /* calculate differences */
        D1_func(B, D1, dimX, dimY, dimZ);
        D2_func(B, D2, dimX, dimY, dimZ);
        D3_func(B, D3, dimX, dimY, dimZ);
        
        /* calculate divergence and image update*/
        TV_main(D1, D2, D3, B, A, lambda, tau, dimX, dimY, dimZ);
        }
    }
    
}

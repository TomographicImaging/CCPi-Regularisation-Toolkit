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

/* ROF-TV denoising/regularization model [1] (2D/3D case)
 * (MEX wrapper for MATLAB)
 * 
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
 * 4. tau - marching step for explicit scheme, ~1 is recommended [REQUIRED]
 *
 * Output:
 * [1] Regularized image/volume 
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 *
 * D. Kazantsev, 2016-18
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter_numb, dimX, dimY, dimZ;
    const int  *dim_array;
    float *Input, *Output=NULL, lambda, tau;
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    Input  = (float *) mxGetData(prhs[0]);
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter_numb =  (int) mxGetScalar(prhs[2]); /* iterations number */
    tau =  (float) mxGetScalar(prhs[3]); /* marching step parameter */  
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if(nrhs != 4) mexErrMsgTxt("Four inputs reqired: Image(2D,3D), regularization parameter, iterations number,  marching step constant");
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        /* output image/volume */
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));                        
    }    
    if (number_of_dims == 3) Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    
    TV_ROF_CPU_main(Input, Output, lambda, iter_numb, tau, dimX, dimY, dimZ);    
}
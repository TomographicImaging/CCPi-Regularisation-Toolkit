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
#include "TV_ROF_GPU_core.h"

/* ROF-TV denoising/regularization model [1] (2D/3D case)
 * (MEX wrapper for MATLAB)
 * 
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED], scalar or the same size as 1
 * 3. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
 * 4. tau - marching step for explicit scheme, ~1 is recommended [REQUIRED]
 * 5. eplsilon: tolerance constant [REQUIRED]
 *
 * Output:
 * [1] Regularized image/volume 
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 *
 * D. Kazantsev, 2016-19
 */
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
     int number_of_dims, iter_numb;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array_i;
    float *Input, *Output=NULL, lambda_scalar, tau, epsil;    
    float *infovec=NULL;
    float *lambda;  
        
    dim_array_i = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    Input  = (float *) mxGetData(prhs[0]);
    /*special check to find the input scalar or an array*/
    int mrows = mxGetM(prhs[1]);
    int ncols = mxGetN(prhs[1]); 
    if (mrows==1 && ncols==1) {        
        lambda = (float*) calloc (1 ,sizeof(float));
        lambda_scalar =  (float) mxGetScalar(prhs[1]); /* regularization parameter */        
        lambda[0] = lambda_scalar;
    }
    else {
        lambda =  (float *) mxGetData(prhs[1]); /* regularization parameter */    
    }
    iter_numb =  (int) mxGetScalar(prhs[2]); /* iterations number */
    tau =  (float) mxGetScalar(prhs[3]); /* marching step parameter */  
    epsil = (float) mxGetScalar(prhs[4]); /* tolerance */  
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if(nrhs != 5) mexErrMsgTxt("Four inputs reqired: Image(2D,3D), regularization parameter, iterations number,  marching step constant, tolerance");
    /*Handling Matlab output data*/
    dimX = dim_array_i[0]; dimY = dim_array_i[1]; dimZ = dim_array_i[2];        
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        /* output image/volume */
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array_i, mxSINGLE_CLASS, mxREAL));          
    }    
    if (number_of_dims == 3) {
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array_i, mxSINGLE_CLASS, mxREAL));
    }
    
    mwSize vecdim[1];
    vecdim[0] = 2;
    infovec = (float*)mxGetPr(plhs[1] = mxCreateNumericArray(1, vecdim, mxSINGLE_CLASS, mxREAL));
    
    if (mrows==1 && ncols==1) {
    TV_ROF_GPU_main(Input, Output, infovec, lambda, 0, iter_numb, tau, epsil, dimX, dimY, dimZ);    
    free(lambda);
    }
    else TV_ROF_GPU_main(Input, Output, infovec, lambda, 1, iter_numb, tau, epsil, dimX, dimY, dimZ);       
}

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
#include "Diffus_4thO_GPU_core.h"

/* CUDA implementation of fourth-order diffusion scheme [1] for piecewise-smooth recovery (2D/3D case)
 * The minimisation is performed using explicit scheme. 
 *
  * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. Edge-preserving parameter (sigma) [REQUIRED]
 * 4. Number of iterations, for explicit scheme >= 150 is recommended [OPTIONAL, default 300]
 * 5. tau - time-marching step for the explicit scheme [OPTIONAL, default 0.015]
 * 6. eplsilon - tolerance constant [OPTIONAL parameter]
 *
 * Output:
 * [1] Regularized image/volume 
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Hajiaboli, M.R., 2011. An anisotropic fourth-order diffusion filter for image noise removal. International Journal of Computer Vision, 92(2), pp.177-191.
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter_numb;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    float *Input, *Output=NULL, lambda, tau, sigma, epsil;
    float *infovec=NULL;
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    Input  = (float *) mxGetData(prhs[0]);
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    sigma = (float) mxGetScalar(prhs[2]); /* Edge-preserving parameter */
    iter_numb = 500; /* iterations number */
    tau = 0.0025; /* marching step parameter */
    epsil = 1.0e-06; /*tolerance parameter*/
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if ((nrhs < 3) || (nrhs > 6)) mexErrMsgTxt("At least 3 parameters is required, all parameters are: Image(2D/3D), Regularisation parameter, Edge-preserving parameter, iterations number, time-marching constant, tolerance");
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6))  iter_numb = (int) mxGetScalar(prhs[3]); /* iterations number */
    if ((nrhs == 5) || (nrhs == 6))  tau =  (float) mxGetScalar(prhs[4]); /* marching step parameter */
    if (nrhs == 6) epsil =  (float) mxGetScalar(prhs[5]); /* tolerance parameter */
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        /* output image/volume */
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    if (number_of_dims == 3) Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    
    mwSize vecdim[1];
    vecdim[0] = 2;
    infovec = (float*)mxGetPr(plhs[1] = mxCreateNumericArray(1, vecdim, mxSINGLE_CLASS, mxREAL));    
    
    Diffus4th_GPU_main(Input, Output, infovec, lambda, sigma, iter_numb, tau, epsil, dimX, dimY, dimZ);
}

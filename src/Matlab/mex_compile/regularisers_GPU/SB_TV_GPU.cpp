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
#include "TV_SB_GPU_core.h"

/* CUDA mex-file for implementation of Split Bregman - TV denoising-regularisation model (2D/3D) [1]
*
* Input Parameters:
* 1. Noisy image/volume
* 2. lambda - regularisation parameter
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon - tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
* 6. print information: 0 (off) or 1 (on)  [OPTIONAL parameter]
*
* Output:
* 1. Filtered/regularized image
*
* This function is based on the Matlab's code and paper by
* [1]. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.
*/

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, methTV, printswitch;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    
    float *Input, *Output=NULL, lambda, epsil;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 6)) mexErrMsgTxt("At least 2 parameters is required, all parameters are: Image(2D/3D), Regularization parameter, Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1'), print switch");
    
    Input  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 100; /* default iterations number */
    epsil = 0.0001; /* default tolerance constant */
    methTV = 0;  /* default isotropic TV penalty */
    printswitch = 0; /*default print is switched, off - 0 */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5) || (nrhs == 6))  iter = (int) mxGetScalar(prhs[2]); /* iterations number */
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6))  epsil =  (float) mxGetScalar(prhs[3]); /* tolerance constant */
    if ((nrhs == 5) || (nrhs == 6))  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
        if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
        if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
        mxFree(penalty_type);
    }
    if (nrhs == 6)  {
        printswitch = (int) mxGetScalar(prhs[5]);
        if ((printswitch != 0) && (printswitch != 1)) mexErrMsgTxt("Print can be enabled by choosing 1 or off - 0");
    }
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    if (number_of_dims == 3) Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    
    /* running the function */
    TV_SB_GPU_main(Input, Output, lambda, iter, epsil, methTV, printswitch, dimX, dimY, dimZ);
}

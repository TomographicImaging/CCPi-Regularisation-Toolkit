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
#include "dTV_FGP_GPU_core.h"

/* CUDA implementation of FGP-dTV [1,2] denoising/regularization model (2D/3D case)
 * which employs structural similarity of the level sets of two images/volumes, see [1,2]
 * The current implementation updates image 1 while image 2 is being fixed.
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. Additional reference image/volume of the same dimensions as (1) [REQUIRED]
 * 3. lambdaPar - regularization parameter [REQUIRED]
 * 4. Number of iterations [OPTIONAL]
 * 5. eplsilon: tolerance constant [OPTIONAL]
 * 6. eta: smoothing constant to calculate gradient of the reference [OPTIONAL] * 
 * 7. TV-type: methodTV - 'iso' (0) or 'l1' (1) [OPTIONAL]
 * 8. nonneg: 'nonnegativity (0 is OFF by default) [OPTIONAL]
 * 9. print information: 0 (off) or 1 (on) [OPTIONAL]
 *
 * Output:
 * [1] Filtered/regularized image/volume
 *
 * This function is based on the Matlab's codes and papers by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 * [2] M. J. Ehrhardt and M. M. Betcke, Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation, SIAM Journal on Imaging Sciences 9(3), pp. 1084–1106
 */
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, methTV, printswitch, nonneg;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    const mwSize *dim_array2;
    
    float *Input, *InputRef, *Output=NULL, lambda, epsil, eta;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    dim_array2 = mxGetDimensions(prhs[1]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 3) || (nrhs > 9)) mexErrMsgTxt("At least 3 parameters is required, all parameters are: Image(2D/3D), Reference(2D/3D), Regularization parameter, iterations number, tolerance, smoothing constant, penalty type ('iso' or 'l1'), nonnegativity switch, print switch");
    
    Input  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    InputRef  = (float *) mxGetData(prhs[1]); /* reference image (2D/3D) */
    lambda =  (float) mxGetScalar(prhs[2]); /* regularization parameter */
    iter = 300; /* default iterations number */
    epsil = 0.0001; /* default tolerance constant */
    eta = 0.01; /* default smoothing constant */
    methTV = 0;  /* default isotropic TV penalty */
    nonneg = 0; /* default nonnegativity switch, off - 0 */
    printswitch = 0; /*default print is switched, off - 0 */
    
        
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    if (number_of_dims == 2) { if ((dimX != dim_array2[0]) || (dimY != dim_array2[1])) mexErrMsgTxt("The input images have different dimensionalities");}
    if (number_of_dims == 3) { if ((dimX != dim_array2[0]) || (dimY != dim_array2[1]) || (dimZ != dim_array2[2])) mexErrMsgTxt("The input volumes have different dimensionalities");}   
    
    
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7) || (nrhs == 8) || (nrhs == 9))  iter = (int) mxGetScalar(prhs[3]); /* iterations number */
    if ((nrhs == 5) || (nrhs == 6) || (nrhs == 7) || (nrhs == 8) || (nrhs == 9))  epsil =  (float) mxGetScalar(prhs[4]); /* tolerance constant */
    if ((nrhs == 6) || (nrhs == 7) || (nrhs == 8) || (nrhs == 9))  {
    eta =  (float) mxGetScalar(prhs[5]); /* smoothing constant for the gradient of InputRef */
    }
    if ((nrhs == 7) || (nrhs == 8) || (nrhs == 9))  {        
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[6]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
        if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
        if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
        mxFree(penalty_type);
    }    
    if ((nrhs == 8) || (nrhs == 9))  {
        nonneg = (int) mxGetScalar(prhs[7]);
        if ((nonneg != 0) && (nonneg != 1)) mexErrMsgTxt("Nonnegativity constraint can be enabled by choosing 1 or off - 0");
    }
    if (nrhs == 9)  {
        printswitch = (int) mxGetScalar(prhs[8]);
        if ((printswitch != 0) && (printswitch != 1)) mexErrMsgTxt("Print can be enabled by choosing 1 or off - 0");
    }    
   
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    if (number_of_dims == 3) Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    
    /* running the function */
    dTV_FGP_GPU_main(Input, InputRef, Output, lambda, iter, epsil, eta, methTV, nonneg, printswitch, dimX, dimY, dimZ);
}
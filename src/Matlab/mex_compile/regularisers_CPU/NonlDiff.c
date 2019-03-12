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
#include "Diffusion_core.h"

/* C-OMP implementation of linear and nonlinear diffusion with the regularisation model [1] (2D/3D case)
 * The minimisation is performed using explicit scheme.
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 4. Number of iterations, for explicit scheme >= 150 is recommended  [OPTIONAL parameter]
 * 5. tau - time-marching step for explicit scheme [OPTIONAL parameter]
 * 6. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight [OPTIONAL parameter]
 * 7. eplsilon - tolerance constant [OPTIONAL parameter]
 *
 * Output:
 * [1] Regularized image/volume 
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter_numb, penaltytype;
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
    iter_numb = 300; /* iterations number */
    tau = 0.025; /* marching step parameter */
    penaltytype = 1; /* Huber penalty by default */
    epsil = 1.0e-06; /*tolerance parameter*/
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if ((nrhs < 3) || (nrhs > 7)) mexErrMsgTxt("At least 3 parameters is required, all parameters are: Image(2D/3D), Regularisation parameter, Edge-preserving parameter, iterations number, time-marching constant, penalty type - Huber, PM or Tukey, tolerance");
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7))  iter_numb = (int) mxGetScalar(prhs[3]); /* iterations number */
    if ((nrhs == 5) || (nrhs == 6) || (nrhs == 7))  tau =  (float) mxGetScalar(prhs[4]); /* marching step parameter */
    if ((nrhs == 6) || (nrhs == 7))  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[5]); /* Huber, PM or Tukey 'Huber' is the default */
        if ((strcmp(penalty_type, "Huber") != 0) && (strcmp(penalty_type, "PM") != 0) && (strcmp(penalty_type, "Tukey") != 0)) mexErrMsgTxt("Choose penalty: 'Huber', 'PM' or 'Tukey',");
        if (strcmp(penalty_type, "Huber") == 0)  penaltytype = 1;  /* enable 'Huber' penalty */
        if (strcmp(penalty_type, "PM") == 0)  penaltytype = 2;  /* enable Perona-Malik penalty */
        if (strcmp(penalty_type, "Tukey") == 0)  penaltytype = 3;  /* enable Tikey Biweight penalty */
        mxFree(penalty_type);
    }    
    if ((nrhs == 7)) epsil =  (float) mxGetScalar(prhs[6]); /* epsilon */
    
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
    
    Diffusion_CPU_main(Input, Output, infovec, lambda, sigma, iter_numb, tau, penaltytype, epsil, dimX, dimY, dimZ);
}

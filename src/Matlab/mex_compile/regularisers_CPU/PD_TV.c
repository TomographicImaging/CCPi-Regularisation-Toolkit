/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2019 Daniil Kazantsev
 * Copyright 2019 Srikanth Nagella, Edoardo Pasca
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
#include "PD_TV_core.h"

/* C-OMP implementation of Primal-Dual TV [1] by Chambolle Pock denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default, 1 is ON)
 * 7. lipschitz_const: convergence related parameter
 * 8. tau: convergence related parameter
 
 * Output:
 * [1] TV - Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * [1] Antonin Chambolle, Thomas Pock. "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", 2010
 */
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, methTV, nonneg;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    float *Input, *infovec=NULL, *Output=NULL, lambda, epsil, lipschitz_const, tau;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 7)) mexErrMsgTxt("At least 2 parameters is required, all parameters are: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1'), nonnegativity switch, lipschitz_const");
    
    Input  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 500; /* default iterations number */
    epsil = 1.0e-06; /* default tolerance constant */
    methTV = 0;  /* default isotropic TV penalty */
    nonneg = 0; /* default nonnegativity switch, off - 0 */
    lipschitz_const = 8.0; /* lipschitz_const */
    tau = 0.0025; /* tau convergence const */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7) || (nrhs == 8))  iter = (int) mxGetScalar(prhs[2]); /* iterations number */
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7) || (nrhs == 8))  epsil =  (float) mxGetScalar(prhs[3]); /* tolerance constant */
    if ((nrhs == 5) || (nrhs == 6) || (nrhs == 7) || (nrhs == 8))  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
        if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
        if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
        mxFree(penalty_type);
    }
    if ((nrhs == 6) || (nrhs == 7) || (nrhs == 8))  {
        nonneg = (int) mxGetScalar(prhs[5]);
        if ((nonneg != 0) && (nonneg != 1)) mexErrMsgTxt("Nonnegativity constraint can be enabled by choosing 1 or off - 0");
    }
    if ((nrhs == 7) || (nrhs == 8)) lipschitz_const = (float) mxGetScalar(prhs[6]);
    if (nrhs == 8) tau = (float) mxGetScalar(prhs[7]);   
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];    
       
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));        
    }
    if (number_of_dims == 3) {    
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    }    
    mwSize vecdim[1];
    vecdim[0] = 2;
    infovec = (float*)mxGetPr(plhs[1] = mxCreateNumericArray(1, vecdim, mxSINGLE_CLASS, mxREAL));
    
    /* running the function */    
    PDTV_CPU_main(Input, Output, infovec, lambda, iter,  epsil, lipschitz_const, methTV, nonneg, tau, dimX, dimY, dimZ);
}

/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazantsev
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
#include "TGV_GPU_core.h"

/* CUDA implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model [1] (2D case only)
 *
 * Input Parameters:
 * 1. Noisy image (2D) (required)
 * 2. lambda - regularisation parameter (required)
 * 3. parameter to control the first-order term (alpha1) (default - 1)
 * 4. parameter to control the second-order term (alpha0) (default - 0.5)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations (default is 300)
 * 6. Lipshitz constant (default is 12)
 *
 * Output:
 * Filtered/regulariaed image 
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter;
    mwSize dimX, dimY;
    const mwSize *dim_array;
    float *Input, *Output=NULL, lambda, alpha0, alpha1, L2;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 6)) mexErrMsgTxt("At least 2 parameters is required, all parameters are: Image(2D), Regularisation parameter, alpha0, alpha1, iterations number, Lipshitz Constant");
    
    Input  = (float *) mxGetData(prhs[0]); /*noisy image (2D) */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularisation parameter */
    alpha1 =  1.0f; /* parameter to control the first-order term */ 
    alpha0 =  0.5f; /* parameter to control the second-order term */
    iter =  300; /* Iterations number */      
    L2 =  12.0f; /* Lipshitz constant */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }   
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5) || (nrhs == 6))  alpha1 =  (float) mxGetScalar(prhs[2]); /* parameter to control the first-order term */ 
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6))  alpha0 =  (float) mxGetScalar(prhs[3]);  /* parameter to control the second-order term */
    if ((nrhs == 5) || (nrhs == 6))  iter =  (int) mxGetScalar(prhs[4]); /* Iterations number */      
    if (nrhs == 6)  L2 =  (float) mxGetScalar(prhs[5]); /* Lipshitz constant */
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1];
    
    if (number_of_dims == 2) {
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        /* running the function */
        TGV_GPU_main(Input, Output, lambda, alpha1, alpha0, iter, L2, dimX, dimY);        
    }
    if (number_of_dims == 3) {mexErrMsgTxt("Only 2D images accepted");}       
}

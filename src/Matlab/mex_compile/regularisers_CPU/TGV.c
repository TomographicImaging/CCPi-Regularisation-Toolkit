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
#include "TGV_core.h"

/* C-OMP implementation of Primal-Dual denoising method for
 * Total Generilized Variation (TGV)-L2 model [1] (2D/3D)
 *
 * Input Parameters:
 * 1. Noisy image/volume (2D/3D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 * 7. eplsilon - tolerance constant [OPTIONAL parameter]
 *
 * Output:
 * [1] Regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])

{
    int number_of_dims, iter;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;

    float *Input, *Output=NULL, lambda, alpha0, alpha1, L2, epsil;
    float *infovec=NULL;

    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);

    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 7)) mexErrMsgTxt("At least 2 parameters is required, all parameters are: Image(2D), Regularisation parameter, alpha0, alpha1, iterations number, Lipshitz Constant, tolerance");

    Input  = (float *) mxGetData(prhs[0]); /*noisy image/volume */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularisation parameter */
    alpha1 =  1.0f; /* parameter to control the first-order term */
    alpha0 =  2.0f; /* parameter to control the second-order term */
    iter =  500; /* Iterations number */
    L2 =  12.0f; /* Lipshitz constant */
    epsil = 1.0e-06; /*tolerance parameter*/

    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7))  alpha1 =  (float) mxGetScalar(prhs[2]); /* parameter to control the first-order term */
    if ((nrhs == 4) || (nrhs == 5) || (nrhs == 6) || (nrhs == 7))  alpha0 =  (float) mxGetScalar(prhs[3]);  /* parameter to control the second-order term */
    if ((nrhs == 5) || (nrhs == 6) || (nrhs == 7))  iter =  (int) mxGetScalar(prhs[4]); /* Iterations number */
    if ((nrhs == 6) || (nrhs == 7))  L2 =  (float) mxGetScalar(prhs[5]); /* Lipshitz constant */
    if (nrhs == 7) epsil =  (float) mxGetScalar(prhs[6]); /* epsilon */

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
    TGV_main(Input, Output, infovec, lambda, alpha1, alpha0, iter, L2, epsil, dimX, dimY, dimZ);
}

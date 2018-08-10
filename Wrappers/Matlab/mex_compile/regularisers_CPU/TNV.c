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
#include "TNV_core.h"
/*
 * C-OMP implementation of Total Nuclear Variation regularisation model (2D + channels) [1]
 * The code is modified from the implementation by Joan Duran <joan.duran@uib.es> see
 * "denoisingPDHG_ipol.cpp" in Joans Collaborative Total Variation package
 *
 * Input Parameters:
 * 1. Noisy volume of 2D + channel dimension, i.e. 3D volume
 * 2. lambda - regularisation parameter
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon - tolerance constant [OPTIONAL parameter]
 * 5. print information: 0 (off) or 1 (on)  [OPTIONAL parameter]
 *
 * Output:
 * 1. Filtered/regularized image
 *
 * [1]. Duran, J., Moeller, M., Sbert, C. and Cremers, D., 2016. Collaborative total variation: a general framework for vectorial TV models. SIAM Journal on Imaging Sciences, 9(1), pp.116-151.
 */
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    float *Input, *Output=NULL, lambda, epsil;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 4)) mexErrMsgTxt("At least 2 parameters is required, all parameters are: Image(2D + channels), Regularisation parameter, Regularization parameter, iterations number, tolerance");
    
    Input  = (float *) mxGetData(prhs[0]); /* noisy sequence of channels (2D + channels) */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 1000; /* default iterations number */
    epsil = 1.00e-05; /* default tolerance constant */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    if ((nrhs == 3) || (nrhs == 4))  iter = (int) mxGetScalar(prhs[2]); /* iterations number */
    if (nrhs == 4)  epsil =  (float) mxGetScalar(prhs[3]); /* tolerance constant */
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    if (number_of_dims == 2) mexErrMsgTxt("The input must be 3D: [X,Y,Channels]");
    if (number_of_dims == 3) {
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        /* running the function */
        TNV_CPU_main(Input, Output, lambda, iter, epsil, dimX, dimY, dimZ);
    }
}
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
#include "utils.h"
/*
 * Function to calculate TV energy value with respect to the denoising variational problem
 * 
 * Input:
 * 1. Denoised Image/volume
 * 2. Original (noisy) Image/volume
 * 3. lambda - regularisation parameter 
 * 
 * Output:
 * 1. Energy function value
 * 
 */
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, dimX, dimY, dimZ, type;
    const int  *dim_array;
    float *Input, *Input0, lambda;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs != 4)) mexErrMsgTxt("4 inputs: Two images or volumes of the same size required, estimated and the original (noisy), regularisation parameter, type");
    
    Input  = (float *) mxGetData(prhs[0]); /* Denoised Image/volume */
    Input0  = (float *) mxGetData(prhs[1]); /* Original (noisy) Image/volume */
    lambda =  (float) mxGetScalar(prhs[2]); /* regularisation parameter */
    type =  (int) mxGetScalar(prhs[3]); /* type of energy */
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    /*output energy function value */
    plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    float *funcvalA = (float *) mxGetData(plhs[0]);
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    if (number_of_dims == 2) {
		TV_energy2D(Input, Input0, funcvalA, lambda, type, dimX, dimY);
		}
    if (number_of_dims == 3) {
        TV_energy3D(Input, Input0, funcvalA, lambda, type, dimX, dimY, dimZ);
    }
}

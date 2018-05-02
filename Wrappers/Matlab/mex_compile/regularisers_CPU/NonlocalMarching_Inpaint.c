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
#include "NonlocalMarching_Inpaint_core.h"

/* C-OMP implementation of Nonlocal Vertical Marching inpainting method (2D case)
 * The method is heuristic but computationally efficent (especially for larger images).
 * It developed specifically to smoothly inpaint horizontal or inclined missing data regions in sinograms
 * The method WILL not work satisfactory if you have lengthy vertical stripes of missing data
 *
 * Input:
 * 1. 2D image or sinogram [REQUIRED]
 * 2. Mask of the same size as A in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data) [REQUIRED]
 * 3. Linear increment to increase searching window size in iterations, values from 1-3 is a good choice [OPTIONAL, default 1]
 * 4. Number of iterations [OPTIONAL, default - calculate based on the mask]
 *
 * Output:
 * 1. Inpainted sinogram  
 * 2. updated mask
 * Reference: TBA
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, dimX, dimY, dimZ, iterations, SW_increment;
    const int  *dim_array;
    const int  *dim_array2;
    float *Input, *Output=NULL;
    unsigned char *Mask, *Mask_upd=NULL;
    
    dim_array = mxGetDimensions(prhs[0]);
    dim_array2 = mxGetDimensions(prhs[1]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    Input  = (float *) mxGetData(prhs[0]);
    Mask  = (unsigned char *) mxGetData(prhs[1]); /* MASK */    
    SW_increment = 1;
    iterations = 0;
            
    if ((nrhs < 2) || (nrhs > 4)) mexErrMsgTxt("At least 4 parameters is required, all parameters are: Image(2D/3D), Mask(2D/3D), Linear increment, Iterations number");
    if ((nrhs == 3) || (nrhs == 4))  SW_increment =  (int) mxGetScalar(prhs[2]); /* linear increment */
    if ((nrhs == 4))  iterations =  (int) mxGetScalar(prhs[3]); /* iterations number */
       
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if (mxGetClassID(prhs[1]) != mxUINT8_CLASS) {mexErrMsgTxt("The mask must be in uint8 precision");}    
    
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        /* output image/volume */
        if ((dimX != dim_array2[0]) || (dimY != dim_array2[1])) mexErrMsgTxt("Input image and the provided mask are of different dimensions!");
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Mask_upd = (unsigned char*)mxGetPr(plhs[1] = mxCreateNumericArray(2, dim_array, mxUINT8_CLASS, mxREAL));
    }    
    if (number_of_dims == 3) {
        mexErrMsgTxt("Currently 2D supported only");        
    }           
    NonlocalMarching_Inpaint_main(Input, Mask, Output, Mask_upd, SW_increment, iterations, 0, dimX, dimY, dimZ);
}
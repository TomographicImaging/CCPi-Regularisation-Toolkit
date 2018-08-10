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
#include "Diffusion_Inpaint_core.h"

/* C-OMP implementation of linear and nonlinear diffusion [1,2] for inpainting task (2D/3D case)
 * The minimisation is performed using explicit scheme. 
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Inpainting Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. lambda - regularization parameter
 * 4. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 5. Number of iterations, for explicit scheme >= 150 is recommended 
 * 6. tau - time-marching step for explicit scheme
 * 7. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
 *
 * Output:
 * [1] Inpainted image/volume 
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 * [2] Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter_numb, penaltytype, i, inpaint_elements;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;   
    const mwSize *dim_array2;   
    
    float *Input, *Output=NULL, lambda, tau, sigma;
    unsigned char *Mask;
    
    dim_array = mxGetDimensions(prhs[0]);
    dim_array2 = mxGetDimensions(prhs[1]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    Input  = (float *) mxGetData(prhs[0]);
    Mask  = (unsigned char *) mxGetData(prhs[1]); /* MASK */
    lambda =  (float) mxGetScalar(prhs[2]); /* regularization parameter */
    sigma = (float) mxGetScalar(prhs[3]); /* Edge-preserving parameter */
    iter_numb = 300; /* iterations number */
    tau = 0.025; /* marching step parameter */
    penaltytype = 1; /* Huber penalty by default */    
  
    if ((nrhs < 4) || (nrhs > 7)) mexErrMsgTxt("At least 4 parameters is required, all parameters are: Image(2D/3D), Mask(2D/3D), Regularisation parameter, Edge-preserving parameter, iterations number, time-marching constant, penalty type - Huber, PM or Tukey");
    if ((nrhs == 5) || (nrhs == 6) || (nrhs == 7))  iter_numb = (int) mxGetScalar(prhs[4]); /* iterations number */
    if ((nrhs == 6) || (nrhs == 7))  tau =  (float) mxGetScalar(prhs[5]); /* marching step parameter */
    if (nrhs == 7)  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[6]); /* Huber, PM or Tukey 'Huber' is the default */
        if ((strcmp(penalty_type, "Huber") != 0) && (strcmp(penalty_type, "PM") != 0) && (strcmp(penalty_type, "Tukey") != 0)) mexErrMsgTxt("Choose penalty: 'Huber', 'PM' or 'Tukey',");
        if (strcmp(penalty_type, "Huber") == 0)  penaltytype = 1;  /* enable 'Huber' penalty */
        if (strcmp(penalty_type, "PM") == 0)  penaltytype = 2;  /* enable Perona-Malik penalty */
        if (strcmp(penalty_type, "Tukey") == 0)  penaltytype = 3;  /* enable Tikey Biweight penalty */
        mxFree(penalty_type);
    }    
    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    if (mxGetClassID(prhs[1]) != mxUINT8_CLASS) {mexErrMsgTxt("The mask must be in uint8 precision");}
    
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    /* output arrays*/
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        /* output image/volume */
        if ((dimX != dim_array2[0]) || (dimY != dim_array2[1])) mexErrMsgTxt("Input image and the provided mask are of different dimensions!");
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }    
    if (number_of_dims == 3) {
        if ((dimX != dim_array2[0]) || (dimY != dim_array2[1]) || (dimZ != dim_array2[2])) mexErrMsgTxt("Input image and the provided mask are of different dimensions!");
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    }    
    
    inpaint_elements = 0;
    for (i=0; i<(int)(dimY*dimX*dimZ); i++) if (Mask[i] == 1) inpaint_elements++;
    if (inpaint_elements == 0) mexErrMsgTxt("The mask is full of zeros, nothing to inpaint");        
    Diffusion_Inpaint_CPU_main(Input, Mask, Output, lambda, sigma, iter_numb, tau, penaltytype, dimX, dimY, dimZ);
}
/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC and Diamond Light Source Ltd. 
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 * Copyright 2018 Diamond Light Source Ltd. 
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
#include "Nonlocal_TV_core.h"

#define EPS 1.0000e-9

/* Matlab wrapper for C-OMP implementation of non-local regulariser
 * Weights and associated indices must be given as an input.
 * Gauss-Seidel fixed point iteration requires ~ 3 iterations, so the main effort
 * goes in pre-calculation of weights and selection of patches
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. AR_i - indeces of i neighbours
 * 3. AR_j - indeces of j neighbours
 * 4. AR_k - indeces of k neighbours (0 - for 2D case)
 * 5. Weights_ij(k) - associated weights 
 * 6. regularisation parameter
 * 7. iterations number 
 
 * Output:
 * 1. denoised image/volume 	
 * Elmoataz, Abderrahim, Olivier Lezoray, and Sébastien Bougleux. "Nonlocal discrete regularization on weighted graphs: a framework for image and manifold processing." IEEE Trans. Image Processing 17, no. 7 (2008): 1047-1060.
 */

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    long number_of_dims,  dimX, dimY, dimZ;
    int IterNumb, NumNeighb = 0;
    unsigned short *H_i, *H_j, *H_k;
    const int  *dim_array;
    const int  *dim_array2;
    float *A_orig, *Output=NULL, *Weights, lambda;
    
    dim_array = mxGetDimensions(prhs[0]);
    dim_array2 = mxGetDimensions(prhs[1]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A_orig  = (float *) mxGetData(prhs[0]); /* a 2D image or a set of 2D images (3D stack) */
    H_i  = (unsigned short *) mxGetData(prhs[1]); /* indeces of i neighbours */
    H_j  = (unsigned short *) mxGetData(prhs[2]); /* indeces of j neighbours */
    H_k  = (unsigned short *) mxGetData(prhs[3]); /* indeces of k neighbours */
    Weights = (float *) mxGetData(prhs[4]); /* weights for patches */
    lambda = (float) mxGetScalar(prhs[5]); /* regularisation parameter */
    IterNumb = (int) mxGetScalar(prhs[6]); /* the number of iterations */
 
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];   
         
    /*****2D INPUT *****/
    if (number_of_dims == 2) {
        dimZ = 0;   
        NumNeighb = dim_array2[2];
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));  
        }
    /*****3D INPUT *****/
    /****************************************************/
    if (number_of_dims == 3) {
        NumNeighb = dim_array2[3];
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    
    /* run the main function here */
    Nonlocal_TV_CPU_main(A_orig, Output, H_i, H_j, H_k, Weights, dimX, dimY, dimZ, NumNeighb, lambda, IterNumb);
}

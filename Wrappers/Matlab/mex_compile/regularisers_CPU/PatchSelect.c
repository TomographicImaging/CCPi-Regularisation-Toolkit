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
#include "PatchSelect_core.h"

/* C-OMP implementation of non-local weight pre-calculation for non-local priors
 * Weights and associated indices are stored into pre-allocated arrays and passed
 * to the regulariser
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. Searching window (half-size of the main bigger searching window, e.g. 11)
 * 3. Similarity window (half-size of the patch window, e.g. 2)
 * 4. The number of neighbours to take (the most prominent after sorting neighbours will be taken)
 * 5. noise-related parameter to calculate non-local weights
 *
 * Output [2D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. Weights_ij - associated weights
 *
 * Output [3D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. AR_k - indeces of j neighbours
 * 4. Weights_ijk - associated weights
 */
/**************************************************/
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int number_of_dims,  SearchWindow, SimilarWin, NumNeighb;
    mwSize dimX, dimY, dimZ;
    unsigned short *H_i=NULL, *H_j=NULL, *H_k=NULL;
    const int  *dim_array;
    float *A, *Weights = NULL, h;
    int dim_array2[3]; /* for 2D data */
    int dim_array3[4]; /* for 3D data */
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /* a 2D or 3D image/volume */
    SearchWindow = (int) mxGetScalar(prhs[1]);    /* Large Searching window */
    SimilarWin = (int) mxGetScalar(prhs[2]);    /* Similarity window (patch-search)*/
    NumNeighb = (int) mxGetScalar(prhs[3]); /* the total number of neighbours to take */
    h = (float) mxGetScalar(prhs[4]); /* NLM parameter */

    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    dim_array2[0] = dimX; dim_array2[1] = dimY; dim_array2[2] = NumNeighb;  /* 2D case */
    dim_array3[0] = dimX; dim_array3[1] = dimY; dim_array3[2] = dimZ; dim_array3[3] = NumNeighb;  /* 3D case */
    
    /****************2D INPUT ***************/
    if (number_of_dims == 2) {
        dimZ = 0;               
        H_i = (unsigned short*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        H_j = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        Weights = (float*)mxGetPr(plhs[2] = mxCreateNumericArray(3, dim_array2, mxSINGLE_CLASS, mxREAL));
        }
    /****************3D INPUT ***************/
    if (number_of_dims == 3) {        
        H_i = (unsigned short*)mxGetPr(plhs[0] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        H_j = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        H_k = (unsigned short*)mxGetPr(plhs[2] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        Weights = (float*)mxGetPr(plhs[3] = mxCreateNumericArray(4, dim_array3, mxSINGLE_CLASS, mxREAL));        
    }
    
    PatchSelect_CPU_main(A, H_i, H_j, H_k, Weights, (long)(dimX), (long)(dimY), (long)(dimZ), SearchWindow, SimilarWin, NumNeighb, h); 
    
 }

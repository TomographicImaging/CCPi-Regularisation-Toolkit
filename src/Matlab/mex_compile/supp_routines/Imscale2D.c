/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC and Diamond Light Source Ltd.
 *
 * Copyright 2019 Daniil Kazantsev
 * Copyright 2019 Srikanth Nagella, Edoardo Pasca
 * Copyright 2019 Diamond Light Source Ltd.
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

/**************************************************/
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int number_of_dims,  X_new, Y_new;
    mwSize dimX, dimY, dimZ;
    const mwSize *dim_array;
    float *A, *B;
    mwSize dim_array2[2]; /* for scaled 2D data */

    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);

    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /* a 2D or 3D image/volume */
    X_new = (int) mxGetScalar(prhs[1]);  /* new size for image */
    Y_new = (int) mxGetScalar(prhs[2]);  /* new size for image */

    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    dim_array2[0] = X_new; dim_array2[1] = Y_new;
    
    B = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array2, mxSINGLE_CLASS, mxREAL));
    
    Im_scale2D(A, B, dimX, dimY, X_new, Y_new);
 }

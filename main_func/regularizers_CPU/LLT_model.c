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
#include "matrix.h"
#include "LLT_model_core.h"

/* C-OMP implementation of Lysaker, Lundervold and Tai (LLT) model of higher order regularization penalty
*
* Input Parameters:
* 1. U0 - origanal noise image/volume
=======
/* C-OMP implementation of Lysaker, Lundervold and Tai (LLT) model of higher order regularization penalty
*
* Input Parameters:
* 1. U0 - original noise image/volume
>>>>>>> fix typo and add include "matrix.h"
* 2. lambda - regularization parameter
* 3. tau - time-step  for explicit scheme
* 4. iter - iterations number
* 5. epsil  - tolerance constant (to terminate earlier)
* 6. switcher - default is 0, switch to (1) to restrictive smoothing in Z dimension (in test)
*
* Output:
* Filtered/regularized image
*
* Example:
* figure;
* Im = double(imread('lena_gray_256.tif'))/255;  % loading image
* u0 = Im + .03*randn(size(Im)); % adding noise
* [Den] = LLT_model(single(u0), 10, 0.1, 1);
*
*
* to compile with OMP support: mex LLT_model.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
* References: Lysaker, Lundervold and Tai (LLT) 2003, IEEE
*
* 28.11.16/Harwell
*/

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, switcher;
    const int  *dim_array;
    float *U0, *U=NULL, *U_old=NULL, *D1=NULL, *D2=NULL, *D3=NULL, lambda, tau, re, re1, epsil, re_old;
    unsigned short *Map=NULL;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    U0  = (float *) mxGetData(prhs[0]); /*origanal noise image/volume*/
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input in single precision is required"); }
    lambda =  (float) mxGetScalar(prhs[1]); /*regularization parameter*/
    tau =  (float) mxGetScalar(prhs[2]); /* time-step */
    iter =  (int) mxGetScalar(prhs[3]); /*iterations number*/
    epsil =  (float) mxGetScalar(prhs[4]); /* tolerance constant */
    switcher =  (int) mxGetScalar(prhs[5]); /*switch on (1) restrictive smoothing in Z dimension*/
     
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1];  dimZ = 1;
    
    if (number_of_dims == 2) {
        /*2D case*/       
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    else if (number_of_dims == 3) {
        /*3D case*/
        dimZ = dim_array[2];
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        if (switcher != 0) {
        Map = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array, mxUINT16_CLASS, mxREAL));       
        }
    }
    else {mexErrMsgTxt("The input data should be 2D or 3D");}
    
    /*Copy U0 to U*/
    copyIm(U0, U, dimX, dimY, dimZ);
    
    count = 1;
    re_old = 0.0f; 
    if (number_of_dims == 2) {
        for(ll = 0; ll < iter; ll++) {
            
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*estimate inner derrivatives */
            der2D(U, D1, D2, dimX, dimY, dimZ);
            /* calculate div^2 and update */
            div_upd2D(U0, U, D1, D2, dimX, dimY, dimZ, lambda, tau);
            
            /* calculate norm to terminate earlier */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(U_old[j] - U[j],2);
                re1 += pow(U_old[j],2);
            }
            re = sqrt(re)/sqrt(re1);           
            if (re < epsil)  count++;
            if (count > 4) break;            
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) break; 
            }
            re_old = re;         
        
        } /*end of iterations*/
        printf("HO iterations stopped at iteration: %i\n", ll);          
    }
    /*3D version*/
    if (number_of_dims == 3) {
        
        if (switcher == 1) {
            /* apply restrictive smoothing */            
            calcMap(U, Map, dimX, dimY, dimZ);
            /*clear outliers */
            cleanMap(Map, dimX, dimY, dimZ);           
        }
        for(ll = 0; ll < iter; ll++) {
            
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*estimate inner derrivatives */
            der3D(U, D1, D2, D3, dimX, dimY, dimZ);          
            /* calculate div^2 and update */
            div_upd3D(U0, U, D1, D2, D3, Map, switcher, dimX, dimY, dimZ, lambda, tau);                 
            
            /* calculate norm to terminate earlier */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(U_old[j] - U[j],2);
                re1 += pow(U_old[j],2);
            }
            re = sqrt(re)/sqrt(re1);           
            if (re < epsil)  count++;
            if (count > 4) break;            
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) break; 
            }
            re_old = re;     
            
        } /*end of iterations*/
        printf("HO iterations stopped at iteration: %i\n", ll);       
    }
}

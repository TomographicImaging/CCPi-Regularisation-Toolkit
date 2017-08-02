/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazanteev
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
#include "SplitBregman_TV_core.h"

/* C-OMP implementation of Split Bregman - TV denoising-regularization model (2D/3D)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon - tolerance constant [OPTIONAL parameter]
 * 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
 *
 * Output:
 * Filtered/regularized image
 *
 * Example:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
 * u = SplitBregman_TV(single(u0), 10, 30, 1e-04);
 *
 * to compile with OMP support: mex SplitBregman_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * References:
 * The Split Bregman Method for L1 Regularized Problems, by Tom Goldstein and Stanley Osher.
 * D. Kazantsev, 2016*
 */


void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, methTV;
    const int  *dim_array;
    float *A, *U=NULL, *U_old=NULL, *Dx=NULL, *Dy=NULL, *Dz=NULL, *Bx=NULL, *By=NULL, *Bz=NULL, lambda, mu, epsil, re, re1, re_old;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    mu =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 35; /* default iterations number */
    epsil = 0.0001; /* default tolerance constant */
    methTV = 0;  /* default isotropic TV penalty */
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5))  iter = (int) mxGetScalar(prhs[2]); /* iterations number */
    if ((nrhs == 4) || (nrhs == 5))  epsil =  (float) mxGetScalar(prhs[3]); /* tolerance constant */
    if (nrhs == 5)  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
        if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
        if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
        mxFree(penalty_type);
    }
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    lambda = 2.0f*mu;
    count = 1;
    re_old = 0.0f;
    /*Handling Matlab output data*/
    dimY = dim_array[0]; dimX = dim_array[1]; dimZ = dim_array[2];
    
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Dx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Dy = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Bx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        By = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        copyIm(A, U, dimX, dimY, dimZ); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /*storing old values*/
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*GS iteration */
            gauss_seidel2D(U, A, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);
            
            if (methTV == 1)  updDxDy_shrinkAniso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
            else updDxDy_shrinkIso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
            
            updBxBy2D(U, Dx, Dy, Bx, By, dimX, dimY);
            
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
            /*printf("%f %i %i \n", re, ll, count); */
            
            /*copyIm(U_old, U, dimX, dimY, dimZ); */
        }
        printf("SB iterations stopped at iteration: %i\n", ll);
    }
    if (number_of_dims == 3) {
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dy = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Bx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        By = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Bz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        
        copyIm(A, U, dimX, dimY, dimZ); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /*storing old values*/
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*GS iteration */
            gauss_seidel3D(U, A, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda, mu);
            
            if (methTV == 1) updDxDyDz_shrinkAniso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            else updDxDyDz_shrinkIso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            
            updBxByBz3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ);
            
            /* calculate norm to terminate earlier */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(U[j] - U_old[j],2);
                re1 += pow(U[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
            if (count > 4) break;
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) break; }
            /*printf("%f %i %i \n", re, ll, count); */
            re_old = re;
        }
        printf("SB iterations stopped at iteration: %i\n", ll);
    }
}
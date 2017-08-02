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

#include "TGV_PD_core.h"
#include "mex.h"

/* C-OMP implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model (2D case only)
 *
 * Input Parameters:
 * 1. Noisy image/volume (2D)
 * 2. lambda - regularization parameter
 * 3. parameter to control first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of CP iterations
 *
 * Output:
 * Filtered/regularized image 
 *
 * Example:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .03*randn(size(Im)); % adding noise
 * tic; u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550); toc;
 *
 * to compile with OMP support: mex TGV_PD.c  TGV_PD_core.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * References:
 * K. Bredies "Total Generalized Variation"
 *
 * 28.11.16/Harwell
 */
 
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll;
    const int  *dim_array;
    float *A, *U, *U_old, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, lambda, L2, tau, sigma,  alpha1, alpha0;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /*origanal noise image/volume*/
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input in single precision is required"); }
    lambda =  (float) mxGetScalar(prhs[1]); /*regularization parameter*/
    alpha1 =  (float) mxGetScalar(prhs[2]); /*first-order term*/
    alpha0 =  (float) mxGetScalar(prhs[3]); /*second-order term*/
    iter =  (int) mxGetScalar(prhs[4]); /*iterations number*/
    if(nrhs != 5) mexErrMsgTxt("Five input parameters is reqired: Image(2D/3D), Regularization parameter, alpha1, alpha0, Iterations");
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1];
    
    if (number_of_dims == 2) {
        /*2D case*/
        dimZ = 1;
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        /*dual variables*/
        P1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        P2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        Q1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Q2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Q3 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        V1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        V1_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        V2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        V2_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));   
        
        
         /*printf("%i \n", i);*/
    L2 = 12.0; /*Lipshitz constant*/
    tau = 1.0/pow(L2,0.5);
    sigma = 1.0/pow(L2,0.5);
    
    /*Copy A to U*/
    copyIm(A, U, dimX, dimY, dimZ);
    
        /* Here primal-dual iterations begin for 2D */
        for(ll = 0; ll < iter; ll++) {
            
            /* Calculate Dual Variable P */
            DualP_2D(U, V1, V2, P1, P2, dimX, dimY, dimZ, sigma);
            
            /*Projection onto convex set for P*/
            ProjP_2D(P1, P2, dimX, dimY, dimZ, alpha1);
            
            /* Calculate Dual Variable Q */
            DualQ_2D(V1, V2, Q1, Q2, Q3, dimX, dimY, dimZ, sigma);
            
            /*Projection onto convex set for Q*/
            ProjQ_2D(Q1, Q2, Q3, dimX, dimY, dimZ, alpha0);
            
            /*saving U into U_old*/
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_2D(U, A, P1, P2, dimX, dimY, dimZ, lambda, tau);
            
            /*get updated solution U*/
            newU(U, U_old, dimX, dimY, dimZ);
            
            /*saving V into V_old*/
            copyIm(V1, V1_old, dimX, dimY, dimZ);
            copyIm(V2, V2_old, dimX, dimY, dimZ);
            
            /* upd V*/
            UpdV_2D(V1, V2, P1, P2, Q1, Q2, Q3, dimX, dimY, dimZ, tau);
            
            /*get new V*/
            newU(V1, V1_old, dimX, dimY, dimZ);
            newU(V2, V2_old, dimX, dimY, dimZ);
        } /*end of iterations*/        
    }
    else if (number_of_dims == 3) {
        mexErrMsgTxt("The input data should be a 2D array");
        /*3D case*/
    }
    else {mexErrMsgTxt("The input data should be a 2D array");}    
   
}

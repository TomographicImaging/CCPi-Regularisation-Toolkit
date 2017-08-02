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
* References:
* The Split Bregman Method for L1 Regularized Problems, by Tom Goldstein and Stanley Osher.
* D. Kazantsev, 2016*
*/


/* 2D-case related Functions */
/*****************************************************************/
float gauss_seidel2D(float *U, float *A, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda, float mu)
{
    float sum, normConst;
    int i,j,i1,i2,j1,j2;
    normConst = 1.0f/(mu + 4.0f*lambda);
    
#pragma omp parallel for shared(U) private(i,j,i1,i2,j1,j2,sum)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            
            sum = Dx[(i2)*dimY + (j)] - Dx[(i)*dimY + (j)] + Dy[(i)*dimY + (j2)] - Dy[(i)*dimY + (j)] - Bx[(i2)*dimY + (j)] + Bx[(i)*dimY + (j)] - By[(i)*dimY + (j2)] + By[(i)*dimY + (j)];
            sum += (U[(i1)*dimY + (j)] + U[(i2)*dimY + (j)] + U[(i)*dimY + (j1)] + U[(i)*dimY + (j2)]);
            sum *= lambda;
            sum += mu*A[(i)*dimY + (j)];
            U[(i)*dimY + (j)] = normConst*sum;
        }}
    return *U;
}

float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1;
    float val1, val11, val2, val22, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(i,j,i1,j1,val1,val11,val2,val22)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            val1 = (U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) + Bx[(i)*dimY + (j)];
            val2 = (U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) + By[(i)*dimY + (j)];
            
            val11 = fabs(val1) - denom_lam; if (val11 < 0) val11 = 0;
            val22 = fabs(val2) - denom_lam; if (val22 < 0) val22 = 0;
            
            if (val1 !=0) Dx[(i)*dimY + (j)] = (val1/fabs(val1))*val11; else Dx[(i)*dimY + (j)] = 0;
            if (val2 !=0) Dy[(i)*dimY + (j)] = (val2/fabs(val2))*val22; else Dy[(i)*dimY + (j)] = 0;
            
        }}
    return 1;
}
float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1;
    float val1, val11, val2, denom, denom_lam;
    denom_lam = 1.0f/lambda;
    
#pragma omp parallel for shared(U,denom_lam) private(i,j,i1,j1,val1,val11,val2,denom)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            val1 = (U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) + Bx[(i)*dimY + (j)];
            val2 = (U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) + By[(i)*dimY + (j)];
            
            denom = sqrt(val1*val1 + val2*val2);
            
            val11 = (denom - denom_lam); if (val11 < 0) val11 = 0.0f;
            
            if (denom != 0.0f) {
                Dx[(i)*dimY + (j)] = val11*(val1/denom);
                Dy[(i)*dimY + (j)] = val11*(val2/denom);
            }
            else {
                Dx[(i)*dimY + (j)] = 0;
                Dy[(i)*dimY + (j)] = 0;
            }
        }}
    return 1;
}
float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY)
{
    int i,j,i1,j1;
#pragma omp parallel for shared(U) private(i,j,i1,j1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            Bx[(i)*dimY + (j)] = Bx[(i)*dimY + (j)] + ((U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) - Dx[(i)*dimY + (j)]);
            By[(i)*dimY + (j)] = By[(i)*dimY + (j)] + ((U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) - Dy[(i)*dimY + (j)]);
        }}
    return 1;
}


/* 3D-case related Functions */
/*****************************************************************/
float gauss_seidel3D(float *U, float *A, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda, float mu)
{
    float normConst, d_val, b_val, sum;
    int i,j,i1,i2,j1,j2,k,k1,k2;
    normConst = 1.0f/(mu + 6.0f*lambda);
#pragma omp parallel for shared(U) private(i,j,i1,i2,j1,j2,k,k1,k2,d_val,b_val,sum)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                i2 = i-1; if (i2 < 0) i2 = i+1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                j2 = j-1; if (j2 < 0) j2 = j+1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                k2 = k-1; if (k2 < 0) k2 = k+1;
                
                d_val = Dx[(dimX*dimY)*k + (i2)*dimY + (j)] - Dx[(dimX*dimY)*k + (i)*dimY + (j)] + Dy[(dimX*dimY)*k + (i)*dimY + (j2)] - Dy[(dimX*dimY)*k + (i)*dimY + (j)] + Dz[(dimX*dimY)*k2 + (i)*dimY + (j)] - Dz[(dimX*dimY)*k + (i)*dimY + (j)];
                b_val = -Bx[(dimX*dimY)*k + (i2)*dimY + (j)] + Bx[(dimX*dimY)*k + (i)*dimY + (j)] - By[(dimX*dimY)*k + (i)*dimY + (j2)] + By[(dimX*dimY)*k + (i)*dimY + (j)] - Bz[(dimX*dimY)*k2 + (i)*dimY + (j)] + Bz[(dimX*dimY)*k + (i)*dimY + (j)];
                sum =  d_val + b_val;
                sum += U[(dimX*dimY)*k + (i1)*dimY + (j)] + U[(dimX*dimY)*k + (i2)*dimY + (j)] + U[(dimX*dimY)*k + (i)*dimY + (j1)] + U[(dimX*dimY)*k + (i)*dimY + (j2)] + U[(dimX*dimY)*k1 + (i)*dimY + (j)] + U[(dimX*dimY)*k2 + (i)*dimY + (j)];
                sum *= lambda;
                sum += mu*A[(dimX*dimY)*k + (i)*dimY + (j)];
                U[(dimX*dimY)*k + (i)*dimY + (j)] = normConst*sum;
            }}}
    return *U;
}

float updDxDyDz_shrinkAniso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda)
{
    int i,j,i1,j1,k,k1,index;
    float val1, val11, val2, val22, val3, val33, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(index,i,j,i1,j1,k,k1,val1,val11,val2,val22,val3,val33)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                index = (dimX*dimY)*k + (i)*dimY + (j);
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[index]) + Bz[index];
                
                val11 = fabs(val1) - denom_lam; if (val11 < 0) val11 = 0;
                val22 = fabs(val2) - denom_lam; if (val22 < 0) val22 = 0;
                val33 = fabs(val3) - denom_lam; if (val33 < 0) val33 = 0;
                
                if (val1 !=0) Dx[index] = (val1/fabs(val1))*val11; else Dx[index] = 0;
                if (val2 !=0) Dy[index] = (val2/fabs(val2))*val22; else Dy[index] = 0;
                if (val3 !=0) Dz[index] = (val3/fabs(val3))*val33; else Dz[index] = 0;
                
            }}}
    return 1;
}
float updDxDyDz_shrinkIso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda)
{
    int i,j,i1,j1,k,k1,index;
    float val1, val11, val2, val3, denom, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(index,denom,i,j,i1,j1,k,k1,val1,val11,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                index = (dimX*dimY)*k + (i)*dimY + (j);
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[index]) + Bz[index];
                
                denom = sqrt(val1*val1 + val2*val2 + val3*val3);
                
                val11 = (denom - denom_lam); if (val11 < 0) val11 = 0.0f;
                
                if (denom != 0.0f) {
                    Dx[index] = val11*(val1/denom);
                    Dy[index] = val11*(val2/denom);
                    Dz[index] = val11*(val3/denom);
                }
                else {
                    Dx[index] = 0;
                    Dy[index] = 0;
                    Dz[index] = 0;
                }               
            }}}
    return 1;
}
float updBxByBz3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ)
{
    int i,j,k,i1,j1,k1;
#pragma omp parallel for shared(U) private(i,j,k,i1,j1,k1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                Bx[(dimX*dimY)*k + (i)*dimY + (j)] = Bx[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dx[(dimX*dimY)*k + (i)*dimY + (j)]);
                By[(dimX*dimY)*k + (i)*dimY + (j)] = By[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dy[(dimX*dimY)*k + (i)*dimY + (j)]);
                Bz[(dimX*dimY)*k + (i)*dimY + (j)] = Bz[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dz[(dimX*dimY)*k + (i)*dimY + (j)]);
                
            }}}
    return 1;
}
/* General Functions */
/*****************************************************************/
/* Copy Image */
float copyIm(float *A, float *B, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, B) private(j)
    for(j=0; j<dimX*dimY*dimZ; j++)  B[j] = A[j];
    return *B;
}
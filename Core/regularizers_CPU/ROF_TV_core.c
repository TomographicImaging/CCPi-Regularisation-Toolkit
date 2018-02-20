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

#include "ROF_TV_core.h"

#define EPS 1.0e-4
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int sign(float x) {
    return (x > 0) - (x < 0);
}

/* C-OMP implementation of ROF-TV denoising/regularization model [1] (2D/3D case)
 *
 * 
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. tau - marching step for explicit scheme, ~1 is recommended [REQUIRED]
 * 4. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
 *
 * Output:
 * [1] Regularized image/volume 
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 *
 * D. Kazantsev, 2016-18
 */

/* Running iterations of TV-ROF function */
float TV_ROF(float *Input, float *Output, int dimX, int dimY, int dimZ, int iterationsNumb, float tau, float lambda)
{
    float *D1, *D2, *D3;
    int i, DimTotal;
    DimTotal = dimX*dimY*dimZ;
    
      
    D1 = (float *) malloc( DimTotal * sizeof(float) );
    D2 = (float *) malloc( DimTotal * sizeof(float) );   
    D3 = (float *) malloc( DimTotal * sizeof(float) );   
	
	   
    /* copy into output */
    copyIm(Input, Output, dimX, dimY, dimZ);
        
    /* start TV iterations */
    for(i=0; i < iterationsNumb; i++) {           
            
            /* calculate differences */
            D1_func(Output, D1, dimX, dimY, dimZ);
            D2_func(Output, D2, dimX, dimY, dimZ);           
            if (dimZ > 1) D3_func(Output, D3, dimX, dimY, dimZ); 			
            TV_kernel(D1, D2, D3, Output, Input, lambda, tau, dimX, dimY, dimZ);  		          
		}           
    free(D1);free(D2); free(D3);
    return *Output;
}

/* calculate differences 1 */
float D1_func(float *A, float *D1, int dimX, int dimY, int dimZ)
{
    float NOMx_1, NOMy_1, NOMy_0, NOMz_1, NOMz_0, denom1, denom2,denom3, T1;
    int i,j,k,i1,i2,k1,j1,j2,k2,index;
    
    if (dimZ > 1) {
#pragma omp parallel for shared (A, D1, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2, NOMx_1,NOMy_1,NOMy_0,NOMz_1,NOMz_0,denom1,denom2,denom3,T1)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                for(k=0; k<dimZ; k++) {
					index = (dimX*dimY)*k + j*dimX+i;
                    /* symmetric boundary conditions (Neuman) */
                    i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                    i2 = i - 1; if (i2 < 0) i2 = i+1;
                    j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                    j2 = j - 1; if (j2 < 0) j2 = j+1;
                    k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                    k2 = k - 1; if (k2 < 0) k2 = k+1;
                    /*B[(dimX*dimY)*k + i*dimY+j] = 0.25*(A[(dimX*dimY)*k + (i1)*dimY + j] + A[(dimX*dimY)*k + (i2)*dimY + j] + A[(dimX*dimY)*k + (i)*dimY + j1] + A[(dimX*dimY)*k + (i)*dimY + j2]) -  A[(dimX*dimY)*k + i*dimY + j];*/
                    
                    /* Forward-backward differences */
                    NOMx_1 = A[(dimX*dimY)*k + j1*dimX + i] - A[index]; /* x+ */
                    NOMy_1 = A[(dimX*dimY)*k + j*dimX + i1] - A[index]; /* y+ */
                    /*NOMx_0 = (A[(i)*dimY + j] - A[(i2)*dimY + j]); */  /* x- */
                    NOMy_0 = A[index] - A[(dimX*dimY)*k + j*dimX + i2]; /* y- */
                    
                    NOMz_1 = A[(dimX*dimY)*k1 + j*dimX + i] - A[index]; /* z+ */
                    NOMz_0 = A[index] - A[(dimX*dimY)*k2 + j*dimX + i]; /* z- */
                    
                    
                    denom1 = NOMx_1*NOMx_1;
                    denom2 = 0.5*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(sign(NOMz_1) + sign(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T1 = sqrt(denom1 + denom2 + denom3 + EPS);
                    D1[index] = NOMx_1/T1;
                }}}
    }
    else {
#pragma omp parallel for shared (A, D1, dimX, dimY) private(i, j, i1, j1, i2, j2,NOMx_1,NOMy_1,NOMy_0,denom1,denom2,T1,index)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
				index = j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1;
                
                /* Forward-backward differences */
                NOMx_1 = A[j1*dimX + i] - A[index]; /* x+ */
                NOMy_1 = A[j*dimX + i1] - A[index]; /* y+ */
                /*NOMx_0 = (A[(i)*dimY + j] - A[(i2)*dimY + j]); */ /* x- */
                NOMy_0 = A[index] - A[(j)*dimX + i2]; /* y- */
                
                denom1 = NOMx_1*NOMx_1;
                denom2 = 0.5*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom2 = denom2*denom2;
                T1 = sqrt(denom1 + denom2 + EPS);
                D1[index] = NOMx_1/T1;
            }}
    }
    return *D1;
}
/* calculate differences 2 */
float D2_func(float *A, float *D2, int dimX, int dimY, int dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2;
    int i,j,k,i1,i2,k1,j1,j2,k2,index;
    
    if (dimZ > 1) {
#pragma omp parallel for shared (A, D2, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2,  NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                for(k=0; k<dimZ; k++) {
					index = (dimX*dimY)*k + j*dimX+i;
                    /* symmetric boundary conditions (Neuman) */
                    i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                    i2 = i - 1; if (i2 < 0) i2 = i+1;
                    j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                    j2 = j - 1; if (j2 < 0) j2 = j+1;
                    k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                    k2 = k - 1; if (k2 < 0) k2 = k+1;
                    
                    
                    /* Forward-backward differences */
                    NOMx_1 = A[(dimX*dimY)*k + (j1)*dimX + i] - A[index]; /* x+ */
                    NOMy_1 = A[(dimX*dimY)*k + (j)*dimX + i1] - A[index]; /* y+ */
                    NOMx_0 = A[index] - A[(dimX*dimY)*k + (j2)*dimX + i]; /* x- */
                    NOMz_1 = A[(dimX*dimY)*k1 + j*dimX + i] - A[index]; /* z+ */
                    NOMz_0 = A[index] - A[(dimX*dimY)*k2 + (j)*dimX + i]; /* z- */
                    
                    
                    denom1 = NOMy_1*NOMy_1;
                    denom2 = 0.5*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(sign(NOMz_1) + sign(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T2 = sqrt(denom1 + denom2 + denom3 + EPS);
                    D2[index] = NOMy_1/T2;
                }}}
    }
    else {
#pragma omp parallel for shared (A, D2, dimX, dimY) private(i, j, i1, j1, i2, j2, NOMx_1,NOMy_1,NOMx_0,denom1,denom2,T2,index)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
				index = j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1;
                
                /* Forward-backward differences */
                NOMx_1 = A[j1*dimX + i] - A[index]; /* x+ */
                NOMy_1 = A[j*dimX + i1] - A[index]; /* y+ */
                NOMx_0 = A[index] - A[j2*dimX + i]; /* x- */
                /*NOMy_0 = A[(i)*dimY + j] - A[(i)*dimY + j2]; */  /* y- */
                
                denom1 = NOMy_1*NOMy_1;
                denom2 = 0.5*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrt(denom1 + denom2 + EPS);
                D2[index] = NOMy_1/T2;
            }}
    }
    return *D2;
}

/* calculate differences 3 */
float D3_func(float *A, float *D3, int dimY, int dimX, int dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMy_0, NOMz_1, denom1, denom2, denom3, T3;
    int index,i,j,k,i1,i2,k1,j1,j2,k2;
    
#pragma omp parallel for shared (A, D3, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2,  NOMx_1, NOMy_1, NOMy_0, NOMx_0, NOMz_1, denom1, denom2, denom3, T3)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            for(k=0; k<dimZ; k++) {
				index = (dimX*dimY)*k + j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1;
                k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                k2 = k - 1; if (k2 < 0) k2 = k+1;
                
                /* Forward-backward differences */
                NOMx_1 = A[(dimX*dimY)*k + (j1)*dimX + i] - A[index]; /* x+ */
                NOMy_1 = A[(dimX*dimY)*k + (j)*dimX + i1] - A[index]; /* y+ */
                NOMy_0 = A[index] - A[(dimX*dimY)*k + (j)*dimX + i2]; /* y- */
                NOMx_0 = A[index] - A[(dimX*dimY)*k + (j2)*dimX + i]; /* x- */
                NOMz_1 = A[(dimX*dimY)*k1 + j*dimX + i] - A[index]; /* z+ */
                /*NOMz_0 = A[(dimX*dimY)*k + (i)*dimY + j] - A[(dimX*dimY)*k2 + (i)*dimY + j]; */ /* z- */
                
                denom1 = NOMz_1*NOMz_1;
                denom2 = 0.5*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                denom3 = 0.5*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom3 = denom3*denom3;
                T3 = sqrt(denom1 + denom2 + denom3 + EPS);
                D3[index] = NOMz_1/T3;
            }}}
    return *D3;
}

/* calculate divergence */
float TV_kernel(float *D1, float *D2, float *D3, float *B, float *A, float lambda, float tau, int dimX, int dimY, int dimZ)
{
    float dv1, dv2, dv3;
    int index,i,j,k,i1,i2,k1,j1,j2,k2;
    
    if (dimZ > 1) {
#pragma omp parallel for shared (D1, D2, D3, B, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2, dv1,dv2,dv3)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                for(k=0; k<dimZ; k++) {
                    index = (dimX*dimY)*k + j*dimX+i;
                    /* symmetric boundary conditions (Neuman) */
                    i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                    i2 = i - 1; if (i2 < 0) i2 = i+1;
                    j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                    j2 = j - 1; if (j2 < 0) j2 = j+1;
                    k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                    k2 = k - 1; if (k2 < 0) k2 = k+1;
                    
                    /*divergence components */
                    dv1 = D1[index] - D1[(dimX*dimY)*k + j2*dimX+i];
                    dv2 = D2[index] - D2[(dimX*dimY)*k + j*dimX+i2];
                    dv3 = D3[index] - D3[(dimX*dimY)*k2 + j*dimX+i];
                    
                    B[index] = B[index] + tau*lambda*(dv1 + dv2 + dv3) + tau*(A[index] - B[index]);
                }}}		
    }
    else {
#pragma omp parallel for shared (D1, D2, B, dimX, dimY) private(index, i, j, i1, j1, i2, j2,dv1,dv2)
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1;
                
                /* divergence components  */
                dv1 = D1[index] - D1[j2*dimX + i];
                dv2 = D2[index] - D2[j*dimX + i2];
                
                //B[index] =  B[index] + tau*lambda*(dv1 + dv2) + tau*(A[index] - B[index]);
                B[index] =  B[index] + tau*((dv1 + dv2) - lambda*(B[index] - A[index]));                
            }}
    }
    return *B;
}

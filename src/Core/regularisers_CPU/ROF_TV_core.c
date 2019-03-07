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

#define EPS 1.0e-8
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*sign function*/
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
 * 5. eplsilon: tolerance constant 
 *
 * Output:
 * [1] Regularized image/volume 
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 */

/* Running iterations of TV-ROF function */
float TV_ROF_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ)
{
    float *D1=NULL, *D2=NULL, *D3=NULL, *Output_prev=NULL;
    float re, re1;
    re = 0.0f; re1 = 0.0f;
    int count = 0;
    int i; 
    long DimTotal,j;
    DimTotal = (long)(dimX*dimY*dimZ);    
    
    D1 = calloc(DimTotal, sizeof(float));
    D2 = calloc(DimTotal, sizeof(float));
    D3 = calloc(DimTotal, sizeof(float));
	   
    /* copy into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    if (epsil != 0.0f) Output_prev = calloc(DimTotal, sizeof(float));
        
    /* start TV iterations */
    for(i=0; i < iterationsNumb; i++) {            
            /* calculate differences */
            D1_func(Output, D1, (long)(dimX), (long)(dimY), (long)(dimZ));
            D2_func(Output, D2, (long)(dimX), (long)(dimY), (long)(dimZ));
            if (dimZ > 1) D3_func(Output, D3, (long)(dimX), (long)(dimY), (long)(dimZ)); 
            TV_kernel(D1, D2, D3, Output, Input, lambdaPar, tau, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /* check early stopping criteria */
            if (epsil != 0.0f) {
            re = 0.0f; re1 = 0.0f;
	            for(j=0; j<DimTotal; j++)
        	    {
        	        re += powf(Output[j] - Output_prev[j],2);
        	        re1 += powf(Output[j],2);
        	    }
              re = sqrtf(re)/sqrtf(re1);
              if (re < epsil)  count++;
              if (count > 4) break;         
            copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            }            
		}           
    free(D1);free(D2); free(D3);
    if (epsil != 0.0f) free(Output_prev); 
    
    /*adding info into info_vector */
    infovector[0] = (float)(i);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */
	
	return 0;
}

/* calculate differences 1 */
float D1_func(float *A, float *D1, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMy_0, NOMz_1, NOMz_0, denom1, denom2,denom3, T1;
    long i,j,k,i1,i2,k1,j1,j2,k2,index;
    
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
                    
                    /* Forward-backward differences */
                    NOMx_1 = A[(dimX*dimY)*k + j1*dimX + i] - A[index]; /* x+ */
                    NOMy_1 = A[(dimX*dimY)*k + j*dimX + i1] - A[index]; /* y+ */
                    /*NOMx_0 = (A[(i)*dimY + j] - A[(i2)*dimY + j]); */  /* x- */
                    NOMy_0 = A[index] - A[(dimX*dimY)*k + j*dimX + i2]; /* y- */
                    
                    NOMz_1 = A[(dimX*dimY)*k1 + j*dimX + i] - A[index]; /* z+ */
                    NOMz_0 = A[index] - A[(dimX*dimY)*k2 + j*dimX + i]; /* z- */
                    
                    
                    denom1 = NOMx_1*NOMx_1;
                    denom2 = 0.5f*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5f*(sign(NOMz_1) + sign(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
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
                denom2 = 0.5f*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom2 = denom2*denom2;
                T1 = sqrtf(denom1 + denom2 + EPS);
                D1[index] = NOMx_1/T1;
            }}
    }
    return *D1;
}
/* calculate differences 2 */
float D2_func(float *A, float *D2, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2;
    long i,j,k,i1,i2,k1,j1,j2,k2,index;
    
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
                    denom2 = 0.5f*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5f*(sign(NOMz_1) + sign(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T2 = sqrtf(denom1 + denom2 + denom3 + EPS);
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
                denom2 = 0.5f*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrtf(denom1 + denom2 + EPS);
                D2[index] = NOMy_1/T2;
            }}
    }
    return *D2;
}

/* calculate differences 3 */
float D3_func(float *A, float *D3, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMy_0, NOMz_1, denom1, denom2, denom3, T3;
    long index,i,j,k,i1,i2,k1,j1,j2,k2;
    
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
                denom2 = 0.5f*(sign(NOMx_1) + sign(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                denom3 = 0.5f*(sign(NOMy_1) + sign(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom3 = denom3*denom3;
                T3 = sqrtf(denom1 + denom2 + denom3 + EPS);
                D3[index] = NOMz_1/T3;
            }}}
    return *D3;
}

/* calculate divergence */
float TV_kernel(float *D1, float *D2, float *D3, float *B, float *A, float lambda, float tau, long dimX, long dimY, long dimZ)
{
    float dv1, dv2, dv3;
    long index,i,j,k,i1,i2,k1,j1,j2,k2;
    
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
                    
                    B[index] += tau*(lambda*(dv1 + dv2 + dv3) - (B[index] - A[index]));   
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

                B[index] += tau*(lambda*(dv1 + dv2) - (B[index] - A[index]));
            }}
    }
    return *B;
}

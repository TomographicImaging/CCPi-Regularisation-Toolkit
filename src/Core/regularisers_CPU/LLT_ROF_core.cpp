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

#include "LLT_ROF_core.h"
#define EPS_LLT 1.0e-12
#define EPS_ROF 1.0e-12
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*sign function*/
int signLLT(float x) {
    return (x > 0) - (x < 0);
}

/* C-OMP implementation of Lysaker, Lundervold and Tai (LLT) model [1] combined with Rudin-Osher-Fatemi [2] TV regularisation penalty.
 *
 * This penalty can deliver visually pleasant piecewise-smooth recovery if regularisation parameters are selected well.
 * The rule of thumb for selection is to start with lambdaLLT = 0 (just the ROF-TV model) and then proceed to increase
 * lambdaLLT starting with smaller values.
 *
 * Input Parameters:
 * 1. U0 - original noise image/volume
 * 2. lambdaROF - ROF-related regularisation parameter
 * 3. lambdaLLT - LLT-related regularisation parameter
 * 4. tau - time-marching step
 * 5. iter - iterations number (for both models)
 * 6. eplsilon: tolerance constant
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * References:
 * [1] Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.
 * [2] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 */

float LLT_ROF_CPU_main(float *Input, float *Output, float *infovector, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ)
{
    long DimTotal;
    int ll, j;
    float re, re1;
    re = 0.0f; re1 = 0.0f;
    int count = 0;
    
    float *D1_LLT=NULL, *D2_LLT=NULL, *D3_LLT=NULL, *D1_ROF=NULL, *D2_ROF=NULL, *D3_ROF=NULL, *Output_prev=NULL;
    DimTotal = (long)(dimX*dimY*dimZ);
    
    D1_ROF = (float*)calloc(DimTotal, sizeof(float));
    D2_ROF = (float*)calloc(DimTotal, sizeof(float));
    D3_ROF = (float*)calloc(DimTotal, sizeof(float));
    
    D1_LLT = (float*)calloc(DimTotal, sizeof(float));
    D2_LLT = (float*)calloc(DimTotal, sizeof(float));
    D3_LLT = (float*)calloc(DimTotal, sizeof(float));
    
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ)); /* initialize  */
    if (epsil != 0.0f) Output_prev = (float*)calloc(DimTotal, sizeof(float));
    
    for(ll = 0; ll < iterationsNumb; ll++) {
        if ((epsil != 0.0f) && (ll % 5 == 0)) copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
        
        if (dimZ == 1) {
            /* 2D case */
            /****************ROF******************/
            /* calculate first-order differences */
            D1_func_ROF(Output, D1_ROF, (long)(dimX), (long)(dimY), 1l);
            D2_func_ROF(Output, D2_ROF, (long)(dimX), (long)(dimY), 1l);
            /****************LLT******************/
            /* estimate second-order derrivatives */
            der2D_LLT(Output, D1_LLT, D2_LLT, (long)(dimX), (long)(dimY), 1l);
            /* Joint update for ROF and LLT models */
            Update2D_LLT_ROF(Input, Output, D1_LLT, D2_LLT, D1_ROF, D2_ROF, lambdaROF, lambdaLLT, tau, (long)(dimX), (long)(dimY), 1l);
        }
        else {
            /* 3D case */
            /* calculate first-order differences */
            D1_func_ROF(Output, D1_ROF, (long)(dimX), (long)(dimY), (long)(dimZ));
            D2_func_ROF(Output, D2_ROF, (long)(dimX), (long)(dimY), (long)(dimZ));
            D3_func_ROF(Output, D3_ROF, (long)(dimX), (long)(dimY), (long)(dimZ));
            /****************LLT******************/
            /* estimate second-order derrivatives */
            der3D_LLT(Output, D1_LLT, D2_LLT, D3_LLT,(long)(dimX), (long)(dimY), (long)(dimZ));
            /* Joint update for ROF and LLT models */
            Update3D_LLT_ROF(Input, Output, D1_LLT, D2_LLT, D3_LLT, D1_ROF, D2_ROF, D3_ROF, lambdaROF, lambdaLLT, tau, (long)(dimX), (long)(dimY), (long)(dimZ));
        }
        
        /* check early stopping criteria */
        if ((epsil != 0.0f) && (ll % 5 == 0)) {
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<DimTotal; j++)
            {
                re += powf(Output[j] - Output_prev[j],2);
                re1 += powf(Output[j],2);
            }
            re = sqrtf(re)/sqrtf(re1);
            if (re < epsil)  count++;
            if (count > 3) break;
        }
        
    } /*end of iterations*/
    free(D1_LLT);free(D2_LLT);free(D3_LLT);
    free(D1_ROF);free(D2_ROF);free(D3_ROF);
    if (epsil != 0.0f) free(Output_prev);
    
    /*adding info into info_vector */
    infovector[0] = (float)(ll);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */
    return 0;
}

/*************************************************************************/
/**********************LLT-related functions *****************************/
/*************************************************************************/
float der2D_LLT(float *U, float *D1, float *D2, long dimX, long dimY, long dimZ)
{
    long i, j, index, i_p, i_m, j_m, j_p;
    float dxx, dyy, denom_xx, denom_yy;
#pragma omp parallel for shared(U,D1,D2) private(i, j, index, i_p, i_m, j_m, j_p, denom_xx, denom_yy, dxx, dyy)
    for (j = 0; j<dimY; j++) {
        for (i = 0; i<dimX; i++) {
            index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            i_p = i + 1; if (i_p == dimX) i_p = i - 1;
            i_m = i - 1; if (i_m < 0) i_m = i + 1;
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            
            dxx = U[j*dimX+i_p] - 2.0f*U[index] + U[j*dimX+i_m];
            dyy = U[j_p*dimX+i] - 2.0f*U[index] + U[j_m*dimX+i];
            
            denom_xx = fabs(dxx) + EPS_LLT;
            denom_yy = fabs(dyy) + EPS_LLT;
            
            D1[index] = dxx / denom_xx;
            D2[index] = dyy / denom_yy;
        }
    }
    return 1;
}

float der3D_LLT(float *U, float *D1, float *D2, float *D3, long dimX, long dimY, long dimZ)
{
    long i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, index;
    float dxx, dyy, dzz, denom_xx, denom_yy, denom_zz;
#pragma omp parallel for shared(U,D1,D2,D3) private(i, j, index, k, i_p, i_m, j_m, j_p, k_p, k_m, denom_xx, denom_yy, denom_zz, dxx, dyy, dzz)
    for (k = 0; k<dimZ; k++) {
        for (j = 0; j<dimY; j++) {
            for (i = 0; i<dimX; i++) {
                /* symmetric boundary conditions (Neuman) */
                i_p = i + 1; if (i_p == dimX) i_p = i - 1;
                i_m = i - 1; if (i_m < 0) i_m = i + 1;
                j_p = j + 1; if (j_p == dimY) j_p = j - 1;
                j_m = j - 1; if (j_m < 0) j_m = j + 1;
                k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
                k_m = k - 1; if (k_m < 0) k_m = k + 1;
                
                index = (dimX*dimY)*k + j*dimX+i;
                
                dxx = U[(dimX*dimY)*k + j*dimX+i_p] - 2.0f*U[index] + U[(dimX*dimY)*k + j*dimX+i_m];
                dyy = U[(dimX*dimY)*k + j_p*dimX+i] - 2.0f*U[index] + U[(dimX*dimY)*k + j_m*dimX+i];
                dzz = U[(dimX*dimY)*k_p + j*dimX+i] - 2.0f*U[index] + U[(dimX*dimY)*k_m + j*dimX+i];
                
                denom_xx = fabs(dxx) + EPS_LLT;
                denom_yy = fabs(dyy) + EPS_LLT;
                denom_zz = fabs(dzz) + EPS_LLT;
                
                D1[index] = dxx / denom_xx;
                D2[index] = dyy / denom_yy;
                D3[index] = dzz / denom_zz;
            }
        }
    }
    return 1;
}

/*************************************************************************/
/**********************ROF-related functions *****************************/
/*************************************************************************/

/* calculate differences 1 */
float D1_func_ROF(float *A, float *D1, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMy_0, NOMz_1, NOMz_0, denom1, denom2,denom3, T1;
    long i,j,k,i1,i2,k1,j1,j2,k2,index;
    
    if (dimZ > 1) {
#pragma omp parallel for shared (A, D1, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2, NOMx_1,NOMy_1,NOMy_0,NOMz_1,NOMz_0,denom1,denom2,denom3,T1)
        for (k = 0; k<dimZ; k++) {
            for (j = 0; j<dimY; j++) {
                for (i = 0; i<dimX; i++) {
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
                    denom2 = 0.5f*(signLLT(NOMy_1) + signLLT(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5f*(signLLT(NOMz_1) + signLLT(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T1 = sqrt(denom1 + denom2 + denom3 + EPS_ROF);
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
                denom2 = 0.5f*(signLLT(NOMy_1) + signLLT(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom2 = denom2*denom2;
                T1 = sqrtf(denom1 + denom2 + EPS_ROF);
                D1[index] = NOMx_1/T1;
            }}
    }
    return *D1;
}
/* calculate differences 2 */
float D2_func_ROF(float *A, float *D2, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2;
    long i,j,k,i1,i2,k1,j1,j2,k2,index;
    
    if (dimZ > 1) {
#pragma omp parallel for shared (A, D2, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2,  NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2)
        for (k = 0; k<dimZ; k++) {
            for (j = 0; j<dimY; j++) {
                for (i = 0; i<dimX; i++) {
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
                    denom2 = 0.5f*(signLLT(NOMx_1) + signLLT(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5f*(signLLT(NOMz_1) + signLLT(NOMz_0))*(MIN(fabs(NOMz_1),fabs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T2 = sqrtf(denom1 + denom2 + denom3 + EPS_ROF);
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
                denom2 = 0.5f*(signLLT(NOMx_1) + signLLT(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrtf(denom1 + denom2 + EPS_ROF);
                D2[index] = NOMy_1/T2;
            }}
    }
    return *D2;
}

/* calculate differences 3 */
float D3_func_ROF(float *A, float *D3, long dimX, long dimY, long dimZ)
{
    float NOMx_1, NOMy_1, NOMx_0, NOMy_0, NOMz_1, denom1, denom2, denom3, T3;
    long index,i,j,k,i1,i2,k1,j1,j2,k2;
    
#pragma omp parallel for shared (A, D3, dimX, dimY, dimZ) private(index, i, j, k, i1, j1, k1, i2, j2, k2,  NOMx_1, NOMy_1, NOMy_0, NOMx_0, NOMz_1, denom1, denom2, denom3, T3)
    for (k = 0; k<dimZ; k++) {
        for (j = 0; j<dimY; j++) {
            for (i = 0; i<dimX; i++) {
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
                denom2 = 0.5f*(signLLT(NOMx_1) + signLLT(NOMx_0))*(MIN(fabs(NOMx_1),fabs(NOMx_0)));
                denom2 = denom2*denom2;
                denom3 = 0.5f*(signLLT(NOMy_1) + signLLT(NOMy_0))*(MIN(fabs(NOMy_1),fabs(NOMy_0)));
                denom3 = denom3*denom3;
                T3 = sqrtf(denom1 + denom2 + denom3 + EPS_ROF);
                D3[index] = NOMz_1/T3;
            }}}
    return *D3;
}

/*************************************************************************/
/**********************ROF-LLT-related functions *************************/
/*************************************************************************/

float Update2D_LLT_ROF(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D1_ROF, float *D2_ROF, float lambdaROF, float lambdaLLT, float tau, long dimX, long dimY, long dimZ)
{
    long i, j, index, i_p, i_m, j_m, j_p;
    float div, laplc, dxx, dyy, dv1, dv2;
#pragma omp parallel for shared(U,U0) private(i, j, index, i_p, i_m, j_m, j_p, laplc, div, dxx, dyy, dv1, dv2)
    for (j = 0; j<dimY; j++) {
        for (i = 0; i<dimX; i++) {
            index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            i_p = i + 1; if (i_p == dimX) i_p = i - 1;
            i_m = i - 1; if (i_m < 0) i_m = i + 1;
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            
            /*LLT-related part*/
            dxx = D1_LLT[j*dimX+i_p] - 2.0f*D1_LLT[index] + D1_LLT[j*dimX+i_m];
            dyy = D2_LLT[j_p*dimX+i] - 2.0f*D2_LLT[index] + D2_LLT[j_m*dimX+i];
            laplc = dxx + dyy; /*build Laplacian*/
            
            /*ROF-related part*/
            dv1 = D1_ROF[index] - D1_ROF[j_m*dimX + i];
            dv2 = D2_ROF[index] - D2_ROF[j*dimX + i_m];
            div = dv1 + dv2; /*build Divirgent*/
            
            /*combine all into one cost function to minimise */
            U[index] += tau*(lambdaROF*(div) - lambdaLLT*(laplc) - (U[index] - U0[index]));
        }
    }
    return *U;
}

float Update3D_LLT_ROF(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D3_LLT, float *D1_ROF, float *D2_ROF, float *D3_ROF, float lambdaROF, float lambdaLLT, float tau, long dimX, long dimY, long dimZ)
{
    long i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, index;
    float div, laplc, dxx, dyy, dzz, dv1, dv2, dv3;
#pragma omp parallel for shared(U,U0) private(i, j, k, index, i_p, i_m, j_m, j_p, k_p, k_m, laplc, div, dxx, dyy, dzz, dv1, dv2, dv3)
    for (k = 0; k<dimZ; k++) {
        for (j = 0; j<dimY; j++) {
            for (i = 0; i<dimX; i++) {
                /* symmetric boundary conditions (Neuman) */
                i_p = i + 1; if (i_p == dimX) i_p = i - 1;
                i_m = i - 1; if (i_m < 0) i_m = i + 1;
                j_p = j + 1; if (j_p == dimY) j_p = j - 1;
                j_m = j - 1; if (j_m < 0) j_m = j + 1;
                k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
                k_m = k - 1; if (k_m < 0) k_m = k + 1;
                
                index = (dimX*dimY)*k + j*dimX+i;
                
                /*LLT-related part*/
                dxx = D1_LLT[(dimX*dimY)*k + j*dimX+i_p] - 2.0f*D1_LLT[index] + D1_LLT[(dimX*dimY)*k + j*dimX+i_m];
                dyy = D2_LLT[(dimX*dimY)*k + j_p*dimX+i] - 2.0f*D2_LLT[index] + D2_LLT[(dimX*dimY)*k + j_m*dimX+i];
                dzz = D3_LLT[(dimX*dimY)*k_p + j*dimX+i] - 2.0f*D3_LLT[index] + D3_LLT[(dimX*dimY)*k_m + j*dimX+i];
                laplc = dxx + dyy + dzz; /*build Laplacian*/
                
                /*ROF-related part*/
                dv1 = D1_ROF[index] - D1_ROF[(dimX*dimY)*k + j_m*dimX+i];
                dv2 = D2_ROF[index] - D2_ROF[(dimX*dimY)*k + j*dimX+i_m];
                dv3 = D3_ROF[index] - D3_ROF[(dimX*dimY)*k_m + j*dimX+i];
                div = dv1 + dv2 + dv3; /*build Divirgent*/
                
                /*combine all into one cost function to minimise */
                U[index] += tau*(lambdaROF*(div) - lambdaLLT*(laplc) - (U[index] - U0[index]));
            }
        }
    }
    return *U;
}

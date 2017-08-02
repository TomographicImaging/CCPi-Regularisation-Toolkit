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
 * References:
 * K. Bredies "Total Generalized Variation"
 *
 * 28.11.16/Harwell
 */
 



/*Calculating dual variable P (using forward differences)*/
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, int dimZ, float sigma)
{
    int i,j;
#pragma omp parallel for shared(U,V1,V2,P1,P2) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[i*dimY + (j)] = P1[i*dimY + (j)] + sigma*((U[(i-1)*dimY + (j)] - U[i*dimY + (j)])  - V1[i*dimY + (j)]);
            else P1[i*dimY + (j)] = P1[i*dimY + (j)] + sigma*((U[(i + 1)*dimY + (j)] - U[i*dimY + (j)])  - V1[i*dimY + (j)]);
            if (j == dimY-1) P2[i*dimY + (j)] = P2[i*dimY + (j)] + sigma*((U[(i)*dimY + (j-1)] - U[i*dimY + (j)])  - V2[i*dimY + (j)]);
            else  P2[i*dimY + (j)] = P2[i*dimY + (j)] + sigma*((U[(i)*dimY + (j+1)] - U[i*dimY + (j)])  - V2[i*dimY + (j)]);
        }}
    return 1;
}
/*Projection onto convex set for P*/
float ProjP_2D(float *P1, float *P2, int dimX, int dimY, int dimZ, float alpha1)
{
    float grad_magn;
    int i,j;
#pragma omp parallel for shared(P1,P2) private(i,j,grad_magn)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            grad_magn = sqrt(pow(P1[i*dimY + (j)],2) + pow(P2[i*dimY + (j)],2));
            grad_magn = grad_magn/alpha1;
            if (grad_magn > 1.0) {
                P1[i*dimY + (j)] = P1[i*dimY + (j)]/grad_magn;
                P2[i*dimY + (j)] = P2[i*dimY + (j)]/grad_magn;
            }
        }}
    return 1;
}
/*Calculating dual variable Q (using forward differences)*/
float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float sigma)
{
    int i,j;
    float q1, q2, q11, q22;
#pragma omp parallel for shared(Q1,Q2,Q3,V1,V2) private(i,j,q1,q2,q11,q22)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1)
            { q1 = (V1[(i-1)*dimY + (j)] - V1[i*dimY + (j)]);
              q11 = (V2[(i-1)*dimY + (j)] - V2[i*dimY + (j)]);
            }
            else {
                q1 = (V1[(i+1)*dimY + (j)] - V1[i*dimY + (j)]);
                q11 = (V2[(i+1)*dimY + (j)] - V2[i*dimY + (j)]);
            }
            if (j == dimY-1) {
                q2 = (V2[(i)*dimY + (j-1)] - V2[i*dimY + (j)]);
                q22 = (V1[(i)*dimY + (j-1)] - V1[i*dimY + (j)]);
            }
            else {
                q2 = (V2[(i)*dimY + (j+1)] - V2[i*dimY + (j)]);
                q22 = (V1[(i)*dimY + (j+1)] - V1[i*dimY + (j)]);
            }
            Q1[i*dimY + (j)] = Q1[i*dimY + (j)] + sigma*(q1);
            Q2[i*dimY + (j)] = Q2[i*dimY + (j)] + sigma*(q2);
            Q3[i*dimY + (j)] = Q3[i*dimY + (j)]  + sigma*(0.5f*(q11 + q22));
        }}
    return 1;
}

float ProjQ_2D(float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float alpha0)
{
    float grad_magn;
    int i,j;
#pragma omp parallel for shared(Q1,Q2,Q3) private(i,j,grad_magn)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            grad_magn = sqrt(pow(Q1[i*dimY + (j)],2) + pow(Q2[i*dimY + (j)],2) + 2*pow(Q3[i*dimY + (j)],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0) {
                Q1[i*dimY + (j)] = Q1[i*dimY + (j)]/grad_magn;
                Q2[i*dimY + (j)] = Q2[i*dimY + (j)]/grad_magn;
                Q3[i*dimY + (j)] = Q3[i*dimY + (j)]/grad_magn;
            }
        }}
    return 1;
}
/* Divergence and projection for P*/
float DivProjP_2D(float *U, float *A, float *P1, float *P2, int dimX, int dimY, int dimZ, float lambda, float tau)
{
    int i,j;
    float P_v1, P_v2, div;
#pragma omp parallel for shared(U,A,P1,P2) private(i,j,P_v1,P_v2,div)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            if (i == 0) P_v1 = (P1[i*dimY + (j)]);
            else P_v1 = (P1[i*dimY + (j)] - P1[(i-1)*dimY + (j)]);
            if (j == 0) P_v2 = (P2[i*dimY + (j)]);
            else  P_v2 = (P2[i*dimY + (j)] - P2[(i)*dimY + (j-1)]);
            div = P_v1 + P_v2;
            U[i*dimY + (j)] = (lambda*(U[i*dimY + (j)] + tau*div) + tau*A[i*dimY + (j)])/(lambda + tau);
        }}
    return *U;
}
/*get updated solution U*/
float newU(float *U, float *U_old, int dimX, int dimY, int dimZ)
{
    int i;
#pragma omp parallel for shared(U,U_old) private(i)
    for(i=0; i<dimX*dimY*dimZ; i++) U[i] = 2*U[i] - U_old[i];
    return *U;
}

/*get update for V*/
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float tau)
{
    int i,j;
    float q1, q11, q2, q22, div1, div2;
#pragma omp parallel for shared(V1,V2,P1,P2,Q1,Q2,Q3) private(i,j, q1, q11, q2, q22, div1, div2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            if (i == 0) {
                q1 = (Q1[i*dimY + (j)]);
                q11 = (Q3[i*dimY + (j)]);
            }
            else {
                q1 = (Q1[i*dimY + (j)] - Q1[(i-1)*dimY + (j)]);
                q11 = (Q3[i*dimY + (j)] - Q3[(i-1)*dimY + (j)]);
            }
            if (j == 0) {
                q2 = (Q2[i*dimY + (j)]);
                q22 = (Q3[i*dimY + (j)]);
            }
            else  {
                q2 = (Q2[i*dimY + (j)] - Q2[(i)*dimY + (j-1)]);
                q22 = (Q3[i*dimY + (j)] - Q3[(i)*dimY + (j-1)]);
            }
            div1 = q1 + q22;
            div2 = q2 + q11;
            V1[i*dimY + (j)] = V1[i*dimY + (j)] + tau*(P1[i*dimY + (j)] + div1);
            V2[i*dimY + (j)] = V2[i*dimY + (j)] + tau*(P2[i*dimY + (j)] + div2);
        }}
    return 1;
}
/* Copy Image */
float copyIm(float *A, float *U, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, U) private(j)
    for(j=0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
    return *U;
}
/*********************3D *********************/

/*Calculating dual variable P (using forward differences)*/
float DualP_3D(float *U, float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ, float sigma)
{
    int i,j,k;
#pragma omp parallel for shared(U,V1,V2,V3,P1,P2,P3) private(i,j,k)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* symmetric boundary conditions (Neuman) */
                if (i == dimX-1) P1[dimX*dimY*k + i*dimY + (j)] = P1[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*k + (i-1)*dimY + (j)] - U[dimX*dimY*k + i*dimY + (j)])  - V1[dimX*dimY*k + i*dimY + (j)]);
                else P1[dimX*dimY*k + i*dimY + (j)] = P1[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*k + (i + 1)*dimY + (j)] - U[dimX*dimY*k + i*dimY + (j)])  - V1[dimX*dimY*k + i*dimY + (j)]);
                if (j == dimY-1) P2[dimX*dimY*k + i*dimY + (j)] = P2[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*k + (i)*dimY + (j-1)] - U[dimX*dimY*k + i*dimY + (j)])  - V2[dimX*dimY*k + i*dimY + (j)]);
                else  P2[dimX*dimY*k + i*dimY + (j)] = P2[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*k + (i)*dimY + (j+1)] - U[dimX*dimY*k + i*dimY + (j)])  - V2[dimX*dimY*k + i*dimY + (j)]);
                if (k == dimZ-1) P3[dimX*dimY*k + i*dimY + (j)] = P3[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*(k-1) + (i)*dimY + (j)] - U[dimX*dimY*k + i*dimY + (j)])  - V3[dimX*dimY*k + i*dimY + (j)]);
                else  P3[dimX*dimY*k + i*dimY + (j)] = P3[dimX*dimY*k + i*dimY + (j)] + sigma*((U[dimX*dimY*(k+1) + (i)*dimY + (j)] - U[dimX*dimY*k + i*dimY + (j)])  - V3[dimX*dimY*k + i*dimY + (j)]);
            }}}
    return 1;
}
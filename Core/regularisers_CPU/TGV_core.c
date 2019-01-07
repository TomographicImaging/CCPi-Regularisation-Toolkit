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

#include "TGV_core.h"

/* C-OMP implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model [1] (2D case only)
 *
 * Input Parameters:
 * 1. Noisy image (2D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 *
 * Output:
 * Filtered/regulariaed image 
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */
 
float TGV_main(float *U0, float *U, float lambda, float alpha1, float alpha0, int iter, float L2, int dimX, int dimY)
{
	long DimTotal;
    int ll;
	float *U_old, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, tau, sigma;
		
		DimTotal = (long)(dimX*dimY);
        
        /* dual variables */
        P1 = calloc(DimTotal, sizeof(float));
        P2 = calloc(DimTotal, sizeof(float));
        
        Q1 = calloc(DimTotal, sizeof(float));
        Q2 = calloc(DimTotal, sizeof(float));
        Q3 = calloc(DimTotal, sizeof(float));
        
        U_old = calloc(DimTotal, sizeof(float));
        
        V1 = calloc(DimTotal, sizeof(float));
        V1_old = calloc(DimTotal, sizeof(float));
        V2 = calloc(DimTotal, sizeof(float));
        V2_old = calloc(DimTotal, sizeof(float));
        
        copyIm(U0, U, (long)(dimX), (long)(dimY), 1l); /* initialize  */
       
        tau = pow(L2,-0.5);
        sigma = pow(L2,-0.5);
    
        /* Primal-dual iterations begin here */
        for(ll = 0; ll < iter; ll++) {
            
            /* Calculate Dual Variable P */
            DualP_2D(U, V1, V2, P1, P2, (long)(dimX), (long)(dimY), sigma);
            
            /*Projection onto convex set for P*/
            ProjP_2D(P1, P2, (long)(dimX), (long)(dimY), alpha1);
            
            /* Calculate Dual Variable Q */
            DualQ_2D(V1, V2, Q1, Q2, Q3, (long)(dimX), (long)(dimY), sigma);
            
            /*Projection onto convex set for Q*/
            ProjQ_2D(Q1, Q2, Q3, (long)(dimX), (long)(dimY), alpha0);
            
            /*saving U into U_old*/
            copyIm(U, U_old, (long)(dimX), (long)(dimY), 1l);
            
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_2D(U, U0, P1, P2, (long)(dimX), (long)(dimY), lambda, tau);
            
            /*get updated solution U*/
            newU(U, U_old, (long)(dimX), (long)(dimY));
            
            /*saving V into V_old*/
            copyIm(V1, V1_old, (long)(dimX), (long)(dimY), 1l);
            copyIm(V2, V2_old, (long)(dimX), (long)(dimY), 1l);
            
            /* upd V*/
            UpdV_2D(V1, V2, P1, P2, Q1, Q2, Q3, (long)(dimX), (long)(dimY), tau);
            
            /*get new V*/
            newU(V1, V1_old, (long)(dimX), (long)(dimY));
            newU(V2, V2_old, (long)(dimX), (long)(dimY));
        } /*end of iterations*/
    /*freeing*/
    free(P1);free(P2);free(Q1);free(Q2);free(Q3);free(U_old);
    free(V1);free(V2);free(V1_old);free(V2_old);
	return *U;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/

/*Calculating dual variable P (using forward differences)*/
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, long dimX, long dimY, float sigma)
{
    long i,j, index;
#pragma omp parallel for shared(U,V1,V2,P1,P2) private(i,j,index)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			 index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[index] += sigma*((U[j*dimX+(i-1)] - U[index]) - V1[index]); 
            else P1[index] += sigma*((U[j*dimX+(i+1)] - U[index])  - V1[index]); 
            if (j == dimY-1) P2[index] += sigma*((U[(j-1)*dimX+i] - U[index])  - V2[index]);
            else  P2[index] += sigma*((U[(j+1)*dimX+i] - U[index])  - V2[index]);
        }}
    return 1;
}
/*Projection onto convex set for P*/
float ProjP_2D(float *P1, float *P2, long dimX, long dimY, float alpha1)
{
    float grad_magn;
    long i,j,index;
#pragma omp parallel for shared(P1,P2) private(i,j,index,grad_magn)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            grad_magn = sqrt(pow(P1[index],2) + pow(P2[index],2));
            grad_magn = grad_magn/alpha1;
            if (grad_magn > 1.0) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
            }
        }}
    return 1;
}
/*Calculating dual variable Q (using forward differences)*/
float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float sigma)
{
    long i,j,index;
    float q1, q2, q11, q22;
#pragma omp parallel for shared(Q1,Q2,Q3,V1,V2) private(i,j,index,q1,q2,q11,q22)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1)
            { q1 = (V1[j*dimX+(i-1)] - V1[index]);
              q11 = (V2[j*dimX+(i-1)] - V2[index]);
            }
            else {
                q1 = (V1[j*dimX+(i+1)] - V1[index]);
                q11 = (V2[j*dimX+(i+1)] - V2[index]);
            }
            if (j == dimY-1) {
                q2 = (V2[(j-1)*dimX+i] - V2[index]);
                q22 = (V1[(j-1)*dimX+i] - V1[index]);
            }
            else {
                q2 = V2[(j+1)*dimX+i] - V2[index];
                q22 = V1[(j+1)*dimX+i] - V1[index];
            }
            Q1[index] += sigma*(q1);
            Q2[index] += sigma*(q2);
            Q3[index] += sigma*(0.5f*(q11 + q22));
        }}
    return 1;
}
float ProjQ_2D(float *Q1, float *Q2, float *Q3, long dimX, long dimY, float alpha0)
{
    float grad_magn;
    long i,j,index;
#pragma omp parallel for shared(Q1,Q2,Q3) private(i,j,index,grad_magn)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            grad_magn = sqrt(pow(Q1[index],2) + pow(Q2[index],2) + 2*pow(Q3[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
            }
        }}
    return 1;
}
/* Divergence and projection for P*/
float DivProjP_2D(float *U, float *U0, float *P1, float *P2, long dimX, long dimY, float lambda, float tau)
{
    long i,j,index;
    float P_v1, P_v2, div;
#pragma omp parallel for shared(U,U0,P1,P2) private(i,j,index,P_v1,P_v2,div)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            if (i == 0) P_v1 = P1[index];
            else P_v1 = P1[index] - P1[j*dimX+(i-1)];
            if (j == 0) P_v2 = P2[index];
            else  P_v2 = P2[index] - P2[(j-1)*dimX+i];
            div = P_v1 + P_v2;
            U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
        }}
    return *U;
}
/*get updated solution U*/
float newU(float *U, float *U_old, long dimX, long dimY)
{
    long i;
#pragma omp parallel for shared(U,U_old) private(i)
    for(i=0; i<dimX*dimY; i++) U[i] = 2*U[i] - U_old[i];
    return *U;
}
/*get update for V*/
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float tau)
{
    long i, j, index;
    float q1, q11, q2, q22, div1, div2;
#pragma omp parallel for shared(V1,V2,P1,P2,Q1,Q2,Q3) private(i, j, index, q1, q11, q2, q22, div1, div2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == 0) {
                q1 = Q1[index];
                q11 = Q3[index];
            }
            else {
                q1 = Q1[index] - Q1[j*dimX+(i-1)];
                q11 = Q3[index] - Q3[j*dimX+(i-1)];
            }
            if (j == 0) {
                q2 = Q2[index];
                q22 = Q3[index];
            }
            else  {
                q2 = Q2[index] - Q2[(j-1)*dimX+i];
                q22 = Q3[index] - Q3[(j-1)*dimX+i];
            }
            div1 = q1 + q22;
            div2 = q2 + q11;
            V1[index] += tau*(P1[index] + div1);
            V2[index] += tau*(P2[index] + div2);
        }}
    return 1;
}

/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2019 Daniil Kazantsev
 * Copyright 2019 Srikanth Nagella, Edoardo Pasca
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

#include "TGV_core.h"

/* C-OMP implementation of Primal-Dual denoising method for
 * Total Generilized Variation (TGV)-L2 model [1] (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume (2D/3D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 * 7. eplsilon: tolerance constant
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 *
 */

float TGV_main(float *U0, float *U, float *infovector, float lambda, float alpha1, float alpha0, int iter, float L2, float epsil, int dimX, int dimY, int dimZ)
{
    long DimTotal;
    int ll, j;
    float re, re1;
    re = 0.0f; re1 = 0.0f;
    int count = 0;
    float *U_old, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, tau, sigma;
    
    DimTotal = (long)(dimX*dimY*dimZ);
    copyIm(U0, U, (long)(dimX), (long)(dimY), (long)(dimZ)); /* initialize */
    tau = pow(L2,-0.5);
    sigma = pow(L2,-0.5);
    
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
    
    if (dimZ == 1) {
        /*2D case*/
        
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
            
            /* check early stopping criteria */
            if ((epsil != 0.0f)  && (ll % 5 == 0)) {
                re = 0.0f; re1 = 0.0f;
                for(j=0; j<DimTotal; j++)
                {
                    re += powf(U[j] - U_old[j],2);
                    re1 += powf(U[j],2);
                }
                re = sqrtf(re)/sqrtf(re1);
                if (re < epsil)  count++;
                if (count > 3) break;
            }
        } /*end of iterations*/
    }
    else {
        /*3D case*/
        float *P3, *Q4, *Q5, *Q6, *V3, *V3_old;
        
        P3 = calloc(DimTotal, sizeof(float));
        Q4 = calloc(DimTotal, sizeof(float));
        Q5 = calloc(DimTotal, sizeof(float));
        Q6 = calloc(DimTotal, sizeof(float));
        V3 = calloc(DimTotal, sizeof(float));
        V3_old = calloc(DimTotal, sizeof(float));
        
        /* Primal-dual iterations begin here */
        for(ll = 0; ll < iter; ll++) {
            
            /* Calculate Dual Variable P */
            DualP_3D(U, V1, V2, V3, P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), sigma);
            
            /*Projection onto convex set for P*/
            ProjP_3D(P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), alpha1);
            
            /* Calculate Dual Variable Q */
            DualQ_3D(V1, V2, V3, Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), sigma);
            
            /*Projection onto convex set for Q*/
            ProjQ_3D(Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), alpha0);
            
            /*saving U into U_old*/
            copyIm(U, U_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_3D(U, U0, P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), lambda, tau);
            
            /*get updated solution U*/
            newU3D(U, U_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /*saving V into V_old*/
            copyIm_3Ar(V1, V2, V3, V1_old, V2_old, V3_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /* upd V*/
            UpdV_3D(V1, V2, V3, P1, P2, P3, Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), tau);
            
            /*get new V*/
            newU3D_3Ar(V1, V2, V3, V1_old, V2_old, V3_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /* check early stopping criteria */
            if ((epsil != 0.0f)  && (ll % 5 == 0)) {
                re = 0.0f; re1 = 0.0f;
                for(j=0; j<DimTotal; j++)
                {
                    re += powf(U[j] - U_old[j],2);
                    re1 += powf(U[j],2);
                }
                re = sqrtf(re)/sqrtf(re1);
                if (re < epsil)  count++;
                if (count > 3) break;
            }
            
        } /*end of iterations*/
        free(P3);free(Q4);free(Q5);free(Q6);free(V3);free(V3_old);
    }
    
    /*freeing*/
    free(P1);free(P2);free(Q1);free(Q2);free(Q3);free(U_old);
    free(V1);free(V2);free(V1_old);free(V2_old);
    
    /*adding info into info_vector */
    infovector[0] = (float)(ll);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */
    
    return 0;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/*Calculating dual variable P (using forward differences)*/
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, long dimX, long dimY, float sigma)
{
    long i,j, index;
#pragma omp parallel for shared(U,V1,V2,P1,P2) private(i,j,index)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[index] += sigma*(-V1[index]);
            else P1[index] += sigma*((U[j*dimX+(i+1)] - U[index])  - V1[index]);
            if (j == dimY-1) P2[index] += sigma*(-V2[index]);
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
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            grad_magn = (sqrtf(P1[index]*P1[index] + P2[index]*P2[index]))/alpha1;
            if (grad_magn > 1.0f) {
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
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            q1 = 0.0f; q11 = 0.0f; q2 = 0.0f; q22 = 0.0f;
            /* boundary conditions (Neuman) */
            if (i != dimX-1){
                q1 = V1[j*dimX+(i+1)] - V1[index];
                q11 = V2[j*dimX+(i+1)] - V2[index];
            }
            if (j != dimY-1) {
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
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            grad_magn = sqrtf(Q1[index]*Q1[index] + Q2[index]*Q2[index] + 2*Q3[index]*Q3[index]);
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0f) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
            }
        }}
    return 1;
}
/* Divergence and projection for P (backward differences)*/
float DivProjP_2D(float *U, float *U0, float *P1, float *P2, long dimX, long dimY, float lambda, float tau)
{
    long i,j,index;
    float P_v1, P_v2, div;
#pragma omp parallel for shared(U,U0,P1,P2) private(i,j,index,P_v1,P_v2,div)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            
            if (i == 0) P_v1 = P1[index];
            else if (i == dimX-1) P_v1 = -P1[j*dimX+(i-1)];
            else P_v1 = P1[index] - P1[j*dimX+(i-1)];
            
            if (j == 0) P_v2 = P2[index];
            else if (j == dimY-1) P_v2 = -P2[(j-1)*dimX+i];
            else P_v2 = P2[index] - P2[(j-1)*dimX+i];
            
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
/*get update for V (backward differences)*/
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float tau)
{
    long i, j, index;
    float q1, q3_x, q3_y, q2, div1, div2;
#pragma omp parallel for shared(V1,V2,P1,P2,Q1,Q2,Q3) private(i, j, index, q1, q3_x, q3_y, q2, div1, div2)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            
            /* boundary conditions (Neuman) */
            if (i == 0) {
                q1 = Q1[index];
                q3_x = Q3[index]; }
            else if (i == dimX-1) {
                q1 = -Q1[j*dimX+(i-1)];
                q3_x = -Q3[j*dimX+(i-1)];  }
            else {
                q1 = Q1[index] - Q1[j*dimX+(i-1)];
                q3_x = Q3[index] - Q3[j*dimX+(i-1)];  }
            
            if (j == 0) {
                q2 = Q2[index];
                q3_y = Q3[index]; }
            else if (j == dimY-1) {
                q2 = -Q2[(j-1)*dimX+i];
                q3_y = -Q3[(j-1)*dimX+i]; }
            else {
                q2 = Q2[index] - Q2[(j-1)*dimX+i];
                q3_y = Q3[index] - Q3[(j-1)*dimX+i]; }
            
            
            div1 = q1 + q3_y;
            div2 = q3_x + q2;
            V1[index] += tau*(P1[index] + div1);
            V2[index] += tau*(P2[index] + div2);
        }}
    return 1;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/*Calculating dual variable P (using forward differences)*/
float DualP_3D(float *U, float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float sigma)
{
    long i,j,k, index;
#pragma omp parallel for shared(U,V1,V2,V3,P1,P2,P3) private(i,j,k,index)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                if (i == dimX-1) P1[index] += sigma*(-V1[index]);
                else P1[index] += sigma*((U[(dimX*dimY)*k + j*dimX+(i+1)] - U[index])  - V1[index]);
                if (j == dimY-1) P2[index] += sigma*(-V2[index]);
                else  P2[index] += sigma*((U[(dimX*dimY)*k + (j+1)*dimX+i] - U[index])  - V2[index]);
                if (k == dimZ-1) P3[index] += sigma*(-V3[index]);
                else  P3[index] += sigma*((U[(dimX*dimY)*(k+1) + j*dimX+i] - U[index])  - V3[index]);
            }}}
    return 1;
}
/*Projection onto convex set for P*/
float ProjP_3D(float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float alpha1)
{
    float grad_magn;
    long i,j,k,index;
#pragma omp parallel for shared(P1,P2,P3) private(i,j,k,index,grad_magn)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                grad_magn = (sqrtf(P1[index]*P1[index] + P2[index]*P2[index]+ P3[index]*P3[index]))/alpha1;
                if (grad_magn > 1.0f) {
                    P1[index] /= grad_magn;
                    P2[index] /= grad_magn;
                    P3[index] /= grad_magn;
                }
            }}}
    return 1;
}
/*Calculating dual variable Q (using forward differences)*/
float DualQ_3D(float *V1, float *V2, float *V3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float sigma)
{
    long i,j,k,index;
    float q1, q2, q3, q11, q22, q33, q44, q55, q66;
#pragma omp parallel for shared(Q1,Q2,Q3,Q4,Q5,Q6,V1,V2,V3) private(i,j,k,index,q1,q2,q3,q11,q22,q33,q44,q55,q66)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                q1 = 0.0f; q11 = 0.0f; q33 = 0.0f; q2 = 0.0f; q22 = 0.0f; q55 = 0.0f; q3 = 0.0f; q44 = 0.0f; q66 = 0.0f;
                /* symmetric boundary conditions (Neuman) */
                if (i != dimX-1){
                    q1 = V1[(dimX*dimY)*k + j*dimX+(i+1)] - V1[index];
                    q11 = V2[(dimX*dimY)*k + j*dimX+(i+1)] - V2[index];
                    q33 = V3[(dimX*dimY)*k + j*dimX+(i+1)] - V3[index];
                }
                if (j != dimY-1) {
                    q2 = V2[(dimX*dimY)*k + (j+1)*dimX+i] - V2[index];
                    q22 = V1[(dimX*dimY)*k + (j+1)*dimX+i] - V1[index];
                    q55 = V3[(dimX*dimY)*k + (j+1)*dimX+i] - V3[index];
                }
                if (k != dimZ-1) {
                    q3 = V3[(dimX*dimY)*(k+1) + j*dimX+i] - V3[index];
                    q44 = V1[(dimX*dimY)*(k+1) + j*dimX+i] - V1[index];
                    q66 = V2[(dimX*dimY)*(k+1) + j*dimX+i] - V2[index];
                }
                
                Q1[index] += sigma*(q1); /*Q11*/
                Q2[index] += sigma*(q2); /*Q22*/
                Q3[index] += sigma*(q3); /*Q33*/
                Q4[index] += sigma*(0.5f*(q11 + q22)); /* Q21 / Q12 */
                Q5[index] += sigma*(0.5f*(q33 + q44)); /* Q31 / Q13 */
                Q6[index] += sigma*(0.5f*(q55 + q66)); /* Q32 / Q23 */
            }}}
    return 1;
}
float ProjQ_3D(float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float alpha0)
{
    float grad_magn;
    long i,j,k,index;
#pragma omp parallel for shared(Q1,Q2,Q3,Q4,Q5,Q6) private(i,j,k,index,grad_magn)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                grad_magn = sqrtf(Q1[index]*Q1[index] + Q2[index]*Q2[index] + Q3[index]*Q3[index] + 2.0f*Q4[index]*Q4[index] + 2.0f*Q5[index]*Q5[index] + 2.0f*Q6[index]*Q6[index]);
                grad_magn = grad_magn/alpha0;
                if (grad_magn > 1.0f) {
                    Q1[index] /= grad_magn;
                    Q2[index] /= grad_magn;
                    Q3[index] /= grad_magn;
                    Q4[index] /= grad_magn;
                    Q5[index] /= grad_magn;
                    Q6[index] /= grad_magn;
                }
            }}}
    return 1;
}
/* Divergence and projection for P*/
float DivProjP_3D(float *U, float *U0, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float lambda, float tau)
{
    long i,j,k,index;
    float P_v1, P_v2, P_v3, div;
#pragma omp parallel for shared(U,U0,P1,P2,P3) private(i,j,k,index,P_v1,P_v2,P_v3,div)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                
                if (i == 0) P_v1 = P1[index];
                else if (i == dimX-1)  P_v1 = -P1[(dimX*dimY)*k + j*dimX+(i-1)];
                else P_v1 = P1[index] - P1[(dimX*dimY)*k + j*dimX+(i-1)];
                if (j == 0) P_v2 = P2[index];
                else if (j == dimY-1) P_v2 = -P2[(dimX*dimY)*k + (j-1)*dimX+i];
                else P_v2 = P2[index] - P2[(dimX*dimY)*k + (j-1)*dimX+i];
                if (k == 0) P_v3 = P3[index];
                else if (k == dimZ-1) P_v3 = -P3[(dimX*dimY)*(k-1) + (j)*dimX+i];
                else P_v3 = P3[index] - P3[(dimX*dimY)*(k-1) + (j)*dimX+i];
                
                div = P_v1 + P_v2 + P_v3;
                U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
            }}}
    return *U;
}
/*get update for V*/
float UpdV_3D(float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float tau)
{
    long i,j,k,index;
    float q1, q4x, q5x, q2, q4y, q6y, q6z, q5z, q3, div1, div2, div3;
#pragma omp parallel for shared(V1,V2,V3,P1,P2,P3,Q1,Q2,Q3,Q4,Q5,Q6) private(i,j,k,index,q1,q4x,q5x,q2,q4y,q6y,q6z,q5z,q3,div1,div2,div3)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                q1 = 0.0f; q4x= 0.0f; q5x= 0.0f; q2= 0.0f; q4y= 0.0f; q6y= 0.0f; q6z= 0.0f; q5z= 0.0f; q3= 0.0f;
                /* Q1 - Q11, Q2 - Q22, Q3 -  Q33, Q4 - Q21/Q12, Q5 - Q31/Q13, Q6 - Q32/Q23*/
                /* symmetric boundary conditions (Neuman) */
                
                if (i == 0) {
                    q1 = Q1[index];
                    q4x = Q4[index];
                    q5x = Q5[index]; }
                else if (i == dimX-1) {
                    q1 = -Q1[(dimX*dimY)*k + j*dimX+(i-1)];
                    q4x = -Q4[(dimX*dimY)*k + j*dimX+(i-1)];
                    q5x = -Q5[(dimX*dimY)*k + j*dimX+(i-1)]; }
                else {
                    q1 = Q1[index] - Q1[(dimX*dimY)*k + j*dimX+(i-1)];
                    q4x = Q4[index] - Q4[(dimX*dimY)*k + j*dimX+(i-1)];
                    q5x = Q5[index] - Q5[(dimX*dimY)*k + j*dimX+(i-1)]; }
                if (j == 0) {
                    q2 = Q2[index];
                    q4y = Q4[index];
                    q6y = Q6[index]; }
                else if (j == dimY-1) {
                    q2 = -Q2[(dimX*dimY)*k + (j-1)*dimX+i];
                    q4y = -Q4[(dimX*dimY)*k + (j-1)*dimX+i];
                    q6y = -Q6[(dimX*dimY)*k + (j-1)*dimX+i]; }
                else {
                    q2 = Q2[index] - Q2[(dimX*dimY)*k + (j-1)*dimX+i];
                    q4y = Q4[index] - Q4[(dimX*dimY)*k + (j-1)*dimX+i];
                    q6y = Q6[index] - Q6[(dimX*dimY)*k + (j-1)*dimX+i]; }
                if (k == 0) {
                    q6z = Q6[index];
                    q5z = Q5[index];
                    q3 = Q3[index]; }
                else if (k == dimZ-1) {
                    q6z = -Q6[(dimX*dimY)*(k-1) + (j)*dimX+i];
                    q5z =  -Q5[(dimX*dimY)*(k-1) + (j)*dimX+i];
                    q3 =  -Q3[(dimX*dimY)*(k-1) + (j)*dimX+i]; }
                else {
                    q6z = Q6[index] - Q6[(dimX*dimY)*(k-1) + (j)*dimX+i];
                    q5z = Q5[index] - Q5[(dimX*dimY)*(k-1) + (j)*dimX+i];
                    q3 = Q3[index] - Q3[(dimX*dimY)*(k-1) + (j)*dimX+i]; }
                
                div1 = q1 + q4y + q5z;
                div2 = q4x + q2 + q6z;
                div3 = q5x + q6y + q3;
                
                V1[index] += tau*(P1[index] + div1);
                V2[index] += tau*(P2[index] + div2);
                V3[index] += tau*(P3[index] + div3);
            }}}
    return 1;
}

float copyIm_3Ar(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, long dimX, long dimY, long dimZ)
{
    long j;
#pragma omp parallel for shared(V1, V2, V3, V1_old, V2_old, V3_old) private(j)
    for (j = 0; j<dimX*dimY*dimZ; j++)  {
        V1_old[j] = V1[j];
        V2_old[j] = V2[j];
        V3_old[j] = V3[j];
    }
    return 1;
}

/*get updated solution U*/
float newU3D(float *U, float *U_old, long dimX, long dimY, long dimZ)
{
    long i;
#pragma omp parallel for shared(U, U_old) private(i)
    for(i=0; i<dimX*dimY*dimZ; i++) U[i] = 2.0f*U[i] - U_old[i];
    return *U;
}


/*get updated solution U*/
float newU3D_3Ar(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, long dimX, long dimY, long dimZ)
{
    long i;
#pragma omp parallel for shared(V1, V2, V3, V1_old, V2_old, V3_old) private(i)
    for(i=0; i<dimX*dimY*dimZ; i++) {
        V1[i] = 2.0f*V1[i] - V1_old[i];
        V2[i] = 2.0f*V2[i] - V2_old[i];
        V3[i] = 2.0f*V3[i] - V3_old[i];
    }
    return 1;
}

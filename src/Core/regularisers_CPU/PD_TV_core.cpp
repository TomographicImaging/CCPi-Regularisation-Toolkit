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

#include "PD_TV_core.h"

/* C-OMP implementation of Primal-Dual TV [1] by Chambolle Pock denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. lipschitz_const: convergence related parameter
 * 6. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 7. nonneg: 'nonnegativity (0 is OFF by default, 1 is ON)

 * Output:
 * [1] TV - Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * [1] Antonin Chambolle, Thomas Pock. "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", 2010
 */

float PDTV_CPU_main(float *Input, float *U, float *infovector, float lambdaPar, int iterationsNumb, float epsil, float lipschitz_const, int methodTV, int nonneg, int dimX, int dimY, int dimZ)
{
    int ll;
    long j, DimTotal;
    float re, re1, sigma, theta, lt, tau;
    re = 0.0f; re1 = 0.0f;
    int count = 0;

    //tau = 1.0/powf(lipschitz_const,0.5);
    //sigma = 1.0/powf(lipschitz_const,0.5);
    tau = lambdaPar*0.1f;
    sigma = 1.0/(lipschitz_const*tau);
    theta = 1.0f;
    lt = tau/lambdaPar;
    ll = 0;


    DimTotal = (long)(dimX*dimY*dimZ);

    copyIm(Input, U, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ <= 1) {
        /*2D case */
        float *U_old=NULL, *P1=NULL, *P2=NULL;

        U_old = (float*)calloc(DimTotal, sizeof(float));
        P1 = (float*)calloc(DimTotal, sizeof(float));
        P2 = (float*)calloc(DimTotal, sizeof(float));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

            /* computing the the dual P variable */
            DualP2D(U, P1, P2, (long)(dimX), (long)(dimY), sigma);

            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (U[j] < 0.0f) U[j] = 0.0f;}

            /* projection step */
            Proj_func2D(P1, P2, methodTV, DimTotal);

            /* copy U to U_old */
            copyIm(U, U_old, (long)(dimX), (long)(dimY), 1l);

            /* calculate divergence */
            DivProj2D(U, Input, P1, P2,(long)(dimX), (long)(dimY), lt, tau);

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
            /*get updated solution*/

            getX(U, U_old, theta, DimTotal);
        }
        free(P1); free(P2); free(U_old);
    }
    else {
          /*3D case*/
        float *U_old=NULL, *P1=NULL, *P2=NULL, *P3=NULL;
        U_old = (float*)calloc(DimTotal, sizeof(float));
        P1 = (float*)calloc(DimTotal, sizeof(float));
        P2 = (float*)calloc(DimTotal, sizeof(float));
        P3 = (float*)calloc(DimTotal, sizeof(float));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

         /* computing the the dual P variable */
            DualP3D(U, P1, P2, P3, (long)(dimX), (long)(dimY),  (long)(dimZ), sigma);

            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (U[j] < 0.0f) U[j] = 0.0f;}

            /* projection step */
            Proj_func3D(P1, P2, P3, methodTV, DimTotal);

            /* copy U to U_old */
            copyIm(U, U_old, (long)(dimX), (long)(dimY), (long)(dimZ));

            DivProj3D(U, Input, P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), lt, tau);

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
            /*get updated solution*/

            getX(U, U_old, theta, DimTotal);
        }
        free(P1); free(P2); free(P3); free(U_old);
    }
    /*adding info into info_vector */
    infovector[0] = (float)(ll);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */

    return 0;
}

/*****************************************************************/
/************************2D-case related Functions */
/*****************************************************************/

/*Calculating dual variable (using forward differences)*/
float DualP2D(float *U, float *P1, float *P2, long dimX, long dimY, float sigma)
{
     long i,j,index;
     #pragma omp parallel for shared(U,P1,P2) private(index,i,j)
     for(j=0; j<dimY; j++) {
       for(i=0; i<dimX; i++) {
          index = j*dimX+i;
          /* symmetric boundary conditions (Neuman) */
          if (i == dimX-1) P1[index] += sigma*(U[j*dimX+(i-1)] - U[index]);
          else P1[index] += sigma*(U[j*dimX+(i+1)] - U[index]);
          if (j == dimY-1) P2[index] += sigma*(U[(j-1)*dimX+i] - U[index]);
          else  P2[index] += sigma*(U[(j+1)*dimX+i] - U[index]);
        }}
     return 1;
}

/* Divergence for P dual */
float DivProj2D(float *U, float *Input, float *P1, float *P2, long dimX, long dimY, float lt, float tau)
{
  long i,j,index;
  float P_v1, P_v2, div_var;
  #pragma omp parallel for shared(U,Input,P1,P2) private(index, i, j, P_v1, P_v2, div_var)
  for(j=0; j<dimY; j++) {
    for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == 0) P_v1 = -P1[index];
            else P_v1 = -(P1[index] - P1[j*dimX+(i-1)]);
            if (j == 0) P_v2 = -P2[index];
            else  P_v2 = -(P2[index] - P2[(j-1)*dimX+i]);
            div_var = P_v1 + P_v2;
            U[index] = (U[index] - tau*div_var + lt*Input[index])/(1.0 + lt);
          }}
  return *U;
}

/*get the updated solution*/
float getX(float *U, float *U_old, float theta, long DimTotal)
{
    long i;
    #pragma omp parallel for shared(U,U_old) private(i)
    for(i=0; i<DimTotal; i++) {
          U[i] +=  theta*(U[i] - U_old[i]);
          }
    return *U;
}


/*****************************************************************/
/************************3D-case related Functions */
/*****************************************************************/
/*Calculating dual variable (using forward differences)*/
float DualP3D(float *U, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float sigma)
{
     long i,j,k,index;
     #pragma omp parallel for shared(U,P1,P2,P3) private(index,i,j,k)
     for(k=0; k<dimZ; k++) {
         for(j=0; j<dimY; j++) {
           for(i=0; i<dimX; i++) {
          index = (dimX*dimY)*k + j*dimX+i;
          /* symmetric boundary conditions (Neuman) */
          if (i == dimX-1) P1[index] += sigma*(U[(dimX*dimY)*k + j*dimX+(i-1)] - U[index]);
          else P1[index] += sigma*(U[(dimX*dimY)*k + j*dimX+(i+1)] - U[index]);
          if (j == dimY-1) P2[index] += sigma*(U[(dimX*dimY)*k + (j-1)*dimX+i] - U[index]);
          else  P2[index] += sigma*(U[(dimX*dimY)*k + (j+1)*dimX+i] - U[index]);
          if (k == dimZ-1) P3[index] += sigma*(U[(dimX*dimY)*(k-1) + j*dimX+i] - U[index]);
          else  P3[index] += sigma*(U[(dimX*dimY)*(k+1) + j*dimX+i] - U[index]);
        }}}
     return 1;
}

/* Divergence for P dual */
float DivProj3D(float *U, float *Input, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float lt, float tau)
{
  long i,j,k,index;
  float P_v1, P_v2, P_v3, div_var;
  #pragma omp parallel for shared(U,Input,P1,P2) private(index, i, j, k, P_v1, P_v2, P_v3, div_var)
  for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = (dimX*dimY)*k + j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == 0) P_v1 = -P1[index];
            else P_v1 = -(P1[index] - P1[(dimX*dimY)*k + j*dimX+(i-1)]);
            if (j == 0) P_v2 = -P2[index];
            else  P_v2 = -(P2[index] - P2[(dimX*dimY)*k + (j-1)*dimX+i]);
            if (k == 0) P_v3 = -P3[index];
            else  P_v3 = -(P3[index] - P3[(dimX*dimY)*(k-1) + j*dimX+i]);
            div_var = P_v1 + P_v2 + P_v3;
            U[index] = (U[index] - tau*div_var + lt*Input[index])/(1.0 + lt);
   }}}
  return *U;
}

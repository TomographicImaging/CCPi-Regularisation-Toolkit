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

#include "FGP_dTV_core.h"
#include <Gradient_calculations.h>

/* C-OMP implementation of FGP-dTV [1,2] denoising/regularization model (2D/3D case)
 * which employs structural similarity of the level sets of two images/volumes, see [1,2]
 * The current implementation updates image 1 while image 2 is being fixed.
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. Additional reference image/volume of the same dimensions as (1) [REQUIRED]
 * 3. lambdaPar - regularization parameter [REQUIRED]
 * 4. Number of iterations [OPTIONAL]
 * 5. eplsilon: tolerance constant [OPTIONAL]
 * 6. eta: smoothing constant to calculate gradient of the reference [OPTIONAL] *
 * 7. TV-type: methodTV - 'iso' (0) or 'l1' (1) [OPTIONAL]
 * 8. nonneg: 'nonnegativity (0 is OFF by default) [OPTIONAL]
 * 9. print information: 0 (off) or 1 (on) [OPTIONAL]
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the Matlab's codes and papers by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 * [2] M. J. Ehrhardt and M. M. Betcke, Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation, SIAM Journal on Imaging Sciences 9(3), pp. 1084–1106
 */

float dTV_FGP_CPU_main(float *Input, float *InputRef, float *Output, float *infovector, float lambdaPar, int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int dimX, int dimY, int dimZ)
{
    int ll;
    long j, DimTotal;
    float re, re1;
    re = 0.0f; re1 = 0.0f;
    float tk = 1.0f;
    float tkp1=1.0f;
    int count = 0;


    float *Output_prev=NULL, *P1=NULL, *P2=NULL, *P1_prev=NULL, *P2_prev=NULL, *R1=NULL, *R2=NULL, *InputRef_x=NULL, *InputRef_y=NULL;
    DimTotal = (long)(dimX*dimY*dimZ);

    if (epsil != 0.0f) Output_prev = (float*)calloc(DimTotal, sizeof(float));
    P1 = (float*)calloc(DimTotal, sizeof(float));
    P2 = (float*)calloc(DimTotal, sizeof(float));
    P1_prev = (float*)calloc(DimTotal, sizeof(float));
    P2_prev = (float*)calloc(DimTotal, sizeof(float));
    R1 = (float*)calloc(DimTotal, sizeof(float));
    R2 = (float*)calloc(DimTotal, sizeof(float));
    InputRef_x = (float*)calloc(DimTotal, sizeof(float));
    InputRef_y = (float*)calloc(DimTotal, sizeof(float));

    if (dimZ <= 1) {
        /*2D case */
        /* calculate gradient field (smoothed) for the reference image */
        GradNorm_func2D(InputRef, InputRef_x, InputRef_y, eta, (long)(dimX), (long)(dimY));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

            if ((epsil != 0.0f)  && (ll % 5 == 0)) copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), 1l);
            /*projects a 2D vector field R-1,2 onto the orthogonal complement of another 2D vector field InputRef_xy*/
            ProjectVect_func2D(R1, R2, InputRef_x, InputRef_y, (long)(dimX), (long)(dimY));

            /* computing the gradient of the objective function */
            Obj_dfunc2D(Input, Output, R1, R2, lambdaPar, (long)(dimX), (long)(dimY));

            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (Output[j] < 0.0f) Output[j] = 0.0f;}

            /*Taking a step towards minus of the gradient*/
            Grad_dfunc2D(P1, P2, Output, R1, R2, InputRef_x, InputRef_y, lambdaPar, (long)(dimX), (long)(dimY));

            /* projection step */
            Proj_func2D(P1, P2, methodTV, DimTotal);

            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_dfunc2D(P1, P1_prev, P2, P2_prev, R1, R2, tkp1, tk, DimTotal);

            copyIm(P1, P1_prev, (long)(dimX), (long)(dimY), 1l);
            copyIm(P2, P2_prev, (long)(dimX), (long)(dimY), 1l);
            tk = tkp1;

            /* check early stopping criteria */
            if ((epsil != 0.0f)  && (ll % 5 == 0)) {
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
        }
    }
    else {
        /*3D case*/
        float *P3=NULL, *P3_prev=NULL, *R3=NULL, *InputRef_z=NULL;

        P3 = (float*)calloc(DimTotal, sizeof(float));
        P3_prev = (float*)calloc(DimTotal, sizeof(float));
        R3 = (float*)calloc(DimTotal, sizeof(float));
        InputRef_z = (float*)calloc(DimTotal, sizeof(float));

        /* calculate gradient field (smoothed) for the reference volume */
        GradNorm_func3D(InputRef, InputRef_x, InputRef_y, InputRef_z, eta, (long)(dimX), (long)(dimY), (long)(dimZ));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

            if ((epsil != 0.0f)  && (ll % 5 == 0)) copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), (long)(dimZ));

            /*projects a 3D vector field R-1,2,3 onto the orthogonal complement of another 3D vector field InputRef_xyz*/
            ProjectVect_func3D(R1, R2, R3, InputRef_x, InputRef_y, InputRef_z, (long)(dimX), (long)(dimY), (long)(dimZ));

            /* computing the gradient of the objective function */
            Obj_dfunc3D(Input, Output, R1, R2, R3, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));

            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (Output[j] < 0.0f) Output[j] = 0.0f;}

            /*Taking a step towards minus of the gradient*/
            Grad_dfunc3D(P1, P2, P3, Output, R1, R2, R3, InputRef_x, InputRef_y, InputRef_z, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));

            /* projection step */
            Proj_func3D(P1, P2, P3, methodTV, DimTotal);

            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_dfunc3D(P1, P1_prev, P2, P2_prev, P3, P3_prev, R1, R2, R3, tkp1, tk, DimTotal);

            /*storing old values*/
            copyIm(P1, P1_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            copyIm(P2, P2_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            copyIm(P3, P3_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            tk = tkp1;

            /* check early stopping criteria */
            if ((epsil != 0.0f)  && (ll % 5 == 0)) {
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
        }

        free(P3); free(P3_prev); free(R3); free(InputRef_z);
    }
    if (epsil != 0.0f) free(Output_prev);
    free(P1); free(P2); free(P1_prev); free(P2_prev); free(R1); free(R2); free(InputRef_x); free(InputRef_y);

    /*adding info into info_vector */
    infovector[0] = (float)(ll);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */

    return 0;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float GradNorm_func2D(float *B, float *B_x, float *B_y, float eta, long dimX, long dimY)
{
    long i,j,index;
    float val1, val2, gradX, gradY, magn;
#pragma omp parallel for shared(B, B_x, B_y) private(i,j,index,val1,val2,gradX,gradY,magn)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* zero boundary conditions */
            if (i == dimX-1) {val1 = 0.0f;} else {val1 = B[j*dimX + (i+1)];}
            if (j == dimY-1) {val2 = 0.0f;} else {val2 = B[(j+1)*dimX + i];}
            gradX = val1 - B[index];
            gradY = val2 - B[index];
            magn = pow(gradX,2) + pow(gradY,2);
            magn = sqrt(magn + pow(eta,2)); /* the eta-smoothed gradients magnitude */
            B_x[index] = gradX/magn;
            B_y[index] = gradY/magn;
        }}
    return 1;
}

float ProjectVect_func2D(float *R1, float *R2, float *B_x, float *B_y, long dimX, long dimY)
{
    long i,j,index;
    float in_prod;
#pragma omp parallel for shared(R1, R2, B_x, B_y) private(index,i,j,in_prod)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            in_prod = R1[index]*B_x[index] + R2[index]*B_y[index];   /* calculate inner product */
            R1[index] = R1[index] - in_prod*B_x[index];
            R2[index] = R2[index] - in_prod*B_y[index];
        }}
    return 1;
}

float Obj_dfunc2D(float *A, float *D, float *R1, float *R2, float lambda, long dimX, long dimY)
{
    float val1, val2;
    long i,j,index;
#pragma omp parallel for shared(A,D,R1,R2) private(index,i,j,val1,val2)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* boundary conditions  */
            if (i == 0) {val1 = 0.0f;} else {val1 = R1[j*dimX + (i-1)];}
            if (j == 0) {val2 = 0.0f;} else {val2 = R2[(j-1)*dimX + i];}
            D[index] = A[index] - lambda*(R1[index] + R2[index] - val1 - val2);
        }}
    return *D;
}
float Grad_dfunc2D(float *P1, float *P2, float *D, float *R1, float *R2, float *B_x, float *B_y, float lambda, long dimX, long dimY)
{
    float val1, val2, multip, in_prod;
    long i,j,index;
    multip = (1.0f/(8.0f*lambda));
#pragma omp parallel for shared(P1,P2,D,R1,R2,B_x,B_y,multip) private(i,j,index,val1,val2,in_prod)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* boundary conditions */
            if (i == dimX-1) val1 = 0.0f; else val1 = D[index] - D[j*dimX + (i+1)];
            if (j == dimY-1) val2 = 0.0f; else val2 = D[index] - D[(j+1)*dimX + i];

            in_prod = val1*B_x[index] + val2*B_y[index];   /* calculate inner product */
            val1 = val1 - in_prod*B_x[index];
            val2 = val2 - in_prod*B_y[index];

            P1[index] = R1[index] + multip*val1;
            P2[index] = R2[index] + multip*val2;

        }}
    return 1;
}
float Rupd_dfunc2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, long DimTotal)
{
    long i;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P1_old,P2_old,R1,R2,multip) private(i)
    for(i=0; i<DimTotal; i++) {
        R1[i] = P1[i] + multip*(P1[i] - P1_old[i]);
        R2[i] = P2[i] + multip*(P2[i] - P2_old[i]);
    }
    return 1;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
float GradNorm_func3D(float *B, float *B_x, float *B_y, float *B_z, float eta, long dimX, long dimY, long dimZ)
{
	GradNorm_3D grad = GradNorm_3D(B, B_x, B_y, B_z, eta, dimX, dimY, dimZ);
	gradient_foward<GradNorm_3D>(&grad);

	return 1;
}

//float GradNorm_func3D(float *B, float *B_x, float *B_y, float *B_z, float eta, long dimX, long dimY, long dimZ)
//{
//    long i, j, k, index;
//    float val1, val2, val3, gradX, gradY, gradZ, magn;
//#pragma omp parallel for shared(B, B_x, B_y, B_z) private(i,j,k,index,val1,val2,val3,gradX,gradY,gradZ,magn)
//    for(k=0; k<dimZ; k++) {
//        for(j=0; j<dimY; j++) {
//            for(i=0; i<dimX; i++) {
//
//                index = (dimX*dimY)*k + j*dimX+i;
//
//                /* zero boundary conditions */
//                if (i == dimX-1) {val1 = 0.0f;} else {val1 = B[(dimX*dimY)*k + j*dimX+(i+1)];}
//                if (j == dimY-1) {val2 = 0.0f;} else {val2 = B[(dimX*dimY)*k + (j+1)*dimX+i];}
//                if (k == dimZ-1) {val3 = 0.0f;} else {val3 = B[(dimX*dimY)*(k+1) + (j)*dimX+i];}
//
//                gradX = val1 - B[index];
//                gradY = val2 - B[index];
//                gradZ = val3 - B[index];
//                magn = pow(gradX,2) + pow(gradY,2) + pow(gradZ,2);
//                magn = sqrt(magn + pow(eta,2)); /* the eta-smoothed gradients magnitude */
//                B_x[index] = gradX/magn;
//                B_y[index] = gradY/magn;
//                B_z[index] = gradZ/magn;
//            }}}
//    return 1;
//}

float ProjectVect_func3D(float *R1, float *R2, float *R3, float *B_x, float *B_y, float *B_z, long dimX, long dimY, long dimZ)
{
    long i,j,k,index;
    float in_prod;
#pragma omp parallel for shared(R1, R2, R3, B_x, B_y, B_z) private(index,i,j,k,in_prod)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                in_prod = R1[index]*B_x[index] + R2[index]*B_y[index] + R3[index]*B_z[index];   /* calculate inner product */
                R1[index] = R1[index] - in_prod*B_x[index];
                R2[index] = R2[index] - in_prod*B_y[index];
                R3[index] = R3[index] - in_prod*B_z[index];
            }}}
    return 1;
}

float Obj_dfunc3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, long dimX, long dimY, long dimZ)
{
    float val1, val2, val3;
    long i,j,k,index;
#pragma omp parallel for shared(A,D,R1,R2,R3) private(index,i,j,k,val1,val2,val3)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                /* boundary conditions */
                if (i == 0) {val1 = 0.0f;} else {val1 = R1[(dimX*dimY)*k + j*dimX + (i-1)];}
                if (j == 0) {val2 = 0.0f;} else {val2 = R2[(dimX*dimY)*k + (j-1)*dimX + i];}
                if (k == 0) {val3 = 0.0f;} else {val3 = R3[(dimX*dimY)*(k-1) + j*dimX + i];}
                D[index] = A[index] - lambda*(R1[index] + R2[index] + R3[index] - val1 - val2 - val3);
            }}}
    return *D;
}
float Grad_dfunc3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float *B_x, float *B_y, float *B_z, float lambda, long dimX, long dimY, long dimZ)
{
    float val1, val2, val3, multip, in_prod;
    long i,j,k, index;
    multip = (1.0f/(26.0f*lambda));
#pragma omp parallel for shared(P1,P2,P3,D,R1,R2,R3,multip) private(index,i,j,k,val1,val2,val3,in_prod)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
                index = (dimX*dimY)*k + j*dimX+i;
                /* boundary conditions */
                if (i == dimX-1) val1 = 0.0f; else val1 = D[index] - D[(dimX*dimY)*k + j*dimX + (i+1)];
                if (j == dimY-1) val2 = 0.0f; else val2 = D[index] - D[(dimX*dimY)*k + (j+1)*dimX + i];
                if (k == dimZ-1) val3 = 0.0f; else val3 = D[index] - D[(dimX*dimY)*(k+1) + j*dimX + i];

                in_prod = val1*B_x[index] + val2*B_y[index] + val3*B_z[index];   /* calculate inner product */
                val1 = val1 - in_prod*B_x[index];
                val2 = val2 - in_prod*B_y[index];
                val3 = val3 - in_prod*B_z[index];

                P1[index] = R1[index] + multip*val1;
                P2[index] = R2[index] + multip*val2;
                P3[index] = R3[index] + multip*val3;
            }}}
    return 1;
}
float Rupd_dfunc3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, long DimTotal)
{
    long i;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P3,P1_old,P2_old,P3_old,R1,R2,R3,multip) private(i)
    for(i=0; i<DimTotal; i++) {
        R1[i] = P1[i] + multip*(P1[i] - P1_old[i]);
        R2[i] = P2[i] + multip*(P2[i] - P2_old[i]);
        R3[i] = P3[i] + multip*(P3[i] - P3_old[i]);
    }
    return 1;
}

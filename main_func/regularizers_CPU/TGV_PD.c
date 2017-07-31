#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

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
 * to compile with OMP support: mex TGV_PD.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * References:
 * K. Bredies "Total Generalized Variation"
 *
 * 28.11.16/Harwell
 */
 
/* 2D functions */
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, int dimZ, float sigma);
float ProjP_2D(float *P1, float *P2, int dimX, int dimY, int dimZ, float alpha1);
float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float sigma);
float ProjQ_2D(float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float alpha0);
float DivProjP_2D(float *U, float *A, float *P1, float *P2, int dimX, int dimY, int dimZ, float lambda, float tau);
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float tau);
/*3D functions*/
float DualP_3D(float *U, float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ, float sigma);

float newU(float *U, float *U_old, int dimX, int dimY, int dimZ);
float copyIm(float *A, float *U, int dimX, int dimY, int dimZ);

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll;
    const int  *dim_array;
    float *A, *U, *U_old, *P1, *P2, *P3, *Q1, *Q2, *Q3, *Q4, *Q5, *Q6, *Q7, *Q8, *Q9, *V1, *V1_old, *V2, *V2_old, *V3, *V3_old, lambda, L2, tau, sigma,  alpha1, alpha0;
    
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
        V2_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));   }
    else if (number_of_dims == 3) {
        mexErrMsgTxt("The input data should be 2D");
        /*3D case*/
//         dimZ = dim_array[2];
//         U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         
//         P1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         P2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         P3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         
//         Q1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q4 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q5 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q6 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q7 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q8 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         Q9 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         
//         U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         
//         V1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         V1_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         V2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         V2_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         V3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
//         V3_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));	  
    }
    else {mexErrMsgTxt("The input data should be 2D");}
    
    
    /*printf("%i \n", i);*/
    L2 = 12.0; /*Lipshitz constant*/
    tau = 1.0/pow(L2,0.5);
    sigma = 1.0/pow(L2,0.5);
    
    /*Copy A to U*/
    copyIm(A, U, dimX, dimY, dimZ);
    
    if (number_of_dims == 2) {
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
    
//     /*3D version*/
//     if (number_of_dims == 3) {
//         /* Here primal-dual iterations begin for 3D */
//         for(ll = 0; ll < iter; ll++) {
//             
//             /* Calculate Dual Variable P */
//             DualP_3D(U, V1, V2, V3, P1, P2, P3, dimX, dimY, dimZ, sigma);
//             
//             /*Projection onto convex set for P*/
//             ProjP_3D(P1, P2, P3, dimX, dimY, dimZ, alpha1);
//             
//             /* Calculate Dual Variable Q */
//             DualQ_3D(V1, V2, V2, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, dimX, dimY, dimZ, sigma);
//             
//         } /*end of iterations*/
//     }
}

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
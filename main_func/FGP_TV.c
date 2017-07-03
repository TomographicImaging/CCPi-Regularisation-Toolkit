#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter [REQUIRED]
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon: tolerance constant [OPTIONAL parameter]
 * 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
 *
 * Output:
 * [1] Filtered/regularized image
 * [2] last function value 
 *
 * Example of image denoising:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .05*randn(size(Im)); % adding noise
 * u = FGP_TV(single(u0), 0.05, 100, 1e-04);
 *
 * to compile with OMP support: mex FGP_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 *
 * D. Kazantsev, 2016-17
 *
 */

float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);
float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY);
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY);

float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ);
float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ);
float Proj_func3D(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ);
float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, int dimX, int dimY, int dimZ);


void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, methTV;
    const int  *dim_array;
    float *A, *D=NULL, *D_old=NULL, *P1=NULL, *P2=NULL, *P3=NULL, *P1_old=NULL, *P2_old=NULL, *P3_old=NULL, *R1=NULL, *R2=NULL, *R3=NULL, lambda, tk, tkp1, re, re1, re_old, epsil, funcval;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");
    
    A  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    lambda =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 50; /* default iterations number */
    epsil = 0.001; /* default tolerance constant */
    methTV = 0;  /* default isotropic TV penalty */
    
    if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5))  iter = (int) mxGetScalar(prhs[2]); /* iterations number */
    if ((nrhs == 4) || (nrhs == 5))  epsil =  (float) mxGetScalar(prhs[3]); /* tolerance constant */
    if (nrhs == 5)  {
        char *penalty_type;
        penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
        if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
        if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
        mxFree(penalty_type);
    }
    /*output function value (last iteration) */
    funcval = 0.0f;
    plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);  
    float *funcvalA = (float *) mxGetData(plhs[1]);
        
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    
    tk = 1.0f;
    tkp1=1.0f;
    count = 1;
    re_old = 0.0f;
    
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        D = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        P1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        P2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        P1_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        P2_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        R1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        R2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        /* begin iterations */
        for(ll=0; ll<iter; ll++) {
            
            /* computing the gradient of the objective function */
            Obj_func2D(A, D, R1, R2, lambda, dimX, dimY);
            
            /*Taking a step towards minus of the gradient*/
            Grad_func2D(P1, P2, D, R1, R2, lambda, dimX, dimY);
            
            /* projection step */
            Proj_func2D(P1, P2, methTV, dimX, dimY);
            
            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func2D(P1, P1_old, P2, P2_old, R1, R2, tkp1, tk, dimX, dimY);
            
            /* calculate norm */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(D[j] - D_old[j],2);
                re1 += pow(D[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
            if (count > 3) {
                Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);            
                funcval = 0.0f; 
                for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
                funcvalA[0] = sqrt(funcval);     
                break; }
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) {
                    Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);            
                    funcval = 0.0f; 
                    for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
                    funcvalA[0] = sqrt(funcval);                     
                    break; }}            
            re_old = re;
            /*printf("%f %i %i \n", re, ll, count); */
            
            /*storing old values*/
            copyIm(D, D_old, dimX, dimY, dimZ);
            copyIm(P1, P1_old, dimX, dimY, dimZ);
            copyIm(P2, P2_old, dimX, dimY, dimZ);
            tk = tkp1;
            
            /* calculating the objective function value */
            if (ll == (iter-1)) {
            Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);            
            funcval = 0.0f; 
            for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
            funcvalA[0] = sqrt(funcval);            
            }
        }
        printf("FGP-TV iterations stopped at iteration %i with the function value %f \n", ll, funcvalA[0]);
    }
    if (number_of_dims == 3) {
        D = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P1_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P2_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        P3_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        R1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        R2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        R3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        
        /* begin iterations */
        for(ll=0; ll<iter; ll++) {
            
            /* computing the gradient of the objective function */
            Obj_func3D(A, D, R1, R2, R3,lambda, dimX, dimY, dimZ);
            
            /*Taking a step towards minus of the gradient*/
            Grad_func3D(P1, P2, P3, D, R1, R2, R3, lambda, dimX, dimY, dimZ);
            
            /* projection step */
            Proj_func3D(P1, P2, P3, dimX, dimY, dimZ);
            
            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func3D(P1, P1_old, P2, P2_old, P3, P3_old, R1, R2, R3, tkp1, tk, dimX, dimY, dimZ);
            
            /* calculate norm - stopping rules*/
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(D[j] - D_old[j],2);
                re1 += pow(D[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            /* stop if the norm residual is less than the tolerance EPS */
            if (re < epsil)  count++;
            if (count > 3) {
                Obj_func3D(A, D, P1, P2, P3,lambda, dimX, dimY, dimZ);
                funcval = 0.0f; 
                for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
                funcvalA[0] = sqrt(funcval);                  
                break;}
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) {
                Obj_func3D(A, D, P1, P2, P3,lambda, dimX, dimY, dimZ);
                funcval = 0.0f; 
                for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
                funcvalA[0] = sqrt(funcval);  
                break; }}
            
            re_old = re;
            /*printf("%f %i %i \n", re, ll, count); */
            
            /*storing old values*/
            copyIm(D, D_old, dimX, dimY, dimZ);
            copyIm(P1, P1_old, dimX, dimY, dimZ);
            copyIm(P2, P2_old, dimX, dimY, dimZ);
            copyIm(P3, P3_old, dimX, dimY, dimZ);
            tk = tkp1;
            
            if (ll == (iter-1)) {
            Obj_func3D(A, D, P1, P2, P3,lambda, dimX, dimY, dimZ);
            funcval = 0.0f; 
            for(j=0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j],2);                           
            funcvalA[0] = sqrt(funcval);            
            }
            
        }
        printf("FGP-TV iterations stopped at iteration %i with the function value %f \n", ll, funcvalA[0]);
    }
}

/* 2D-case related Functions */
/*****************************************************************/
float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
    float val1, val2;
    int i,j;
#pragma omp parallel for shared(A,D,R1,R2) private(i,j,val1,val2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* boundary conditions  */
            if (i == 0) {val1 = 0.0f;} else {val1 = R1[(i-1)*dimY + (j)];}
            if (j == 0) {val2 = 0.0f;} else {val2 = R2[(i)*dimY + (j-1)];}
            D[(i)*dimY + (j)] = A[(i)*dimY + (j)] - lambda*(R1[(i)*dimY + (j)] + R2[(i)*dimY + (j)] - val1 - val2);
        }}
    return *D;
}
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
    float val1, val2, multip;
    int i,j;
    multip = (1.0f/(8.0f*lambda));
#pragma omp parallel for shared(P1,P2,D,R1,R2,multip) private(i,j,val1,val2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* boundary conditions */
            if (i == dimX-1) val1 = 0.0f; else val1 = D[(i)*dimY + (j)] - D[(i+1)*dimY + (j)];
            if (j == dimY-1) val2 = 0.0f; else val2 = D[(i)*dimY + (j)] - D[(i)*dimY + (j+1)];
            P1[(i)*dimY + (j)] = R1[(i)*dimY + (j)] + multip*val1;
            P2[(i)*dimY + (j)] = R2[(i)*dimY + (j)] + multip*val2;
        }}
    return 1;
}
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY)
{
    float val1, val2, denom;
    int i,j;
    if (methTV == 0) {
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,denom)
        for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {
                denom = pow(P1[(i)*dimY + (j)],2) +  pow(P2[(i)*dimY + (j)],2);
                if (denom > 1) {
                    P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)]/sqrt(denom);
                    P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)]/sqrt(denom);
                }
            }}
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,val1,val2)
        for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {
                val1 = fabs(P1[(i)*dimY + (j)]);
                val2 = fabs(P2[(i)*dimY + (j)]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)]/val1;
                P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)]/val2;
            }}
    }
    return 1;
}
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY)
{
    int i,j;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P1_old,P2_old,R1,R2,multip) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            R1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] + multip*(P1[(i)*dimY + (j)] - P1_old[(i)*dimY + (j)]);
            R2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] + multip*(P2[(i)*dimY + (j)] - P2_old[(i)*dimY + (j)]);
        }}
    return 1;
}

/* 3D-case related Functions */
/*****************************************************************/
float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ)
{
    float val1, val2, val3;
    int i,j,k;
#pragma omp parallel for shared(A,D,R1,R2,R3) private(i,j,k,val1,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* boundary conditions */
                if (i == 0) {val1 = 0.0f;} else {val1 = R1[(dimX*dimY)*k + (i-1)*dimY + (j)];}
                if (j == 0) {val2 = 0.0f;} else {val2 = R2[(dimX*dimY)*k + (i)*dimY + (j-1)];}
                if (k == 0) {val3 = 0.0f;} else {val3 = R3[(dimX*dimY)*(k-1) + (i)*dimY + (j)];}
                D[(dimX*dimY)*k + (i)*dimY + (j)] = A[(dimX*dimY)*k + (i)*dimY + (j)] - lambda*(R1[(dimX*dimY)*k + (i)*dimY + (j)] + R2[(dimX*dimY)*k + (i)*dimY + (j)] + R3[(dimX*dimY)*k + (i)*dimY + (j)] - val1 - val2 - val3);
            }}}
    return *D;
}
float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ)
{
    float val1, val2, val3, multip;
    int i,j,k;
    multip = (1.0f/(8.0f*lambda));
#pragma omp parallel for shared(P1,P2,P3,D,R1,R2,R3,multip) private(i,j,k,val1,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* boundary conditions */
                if (i == dimX-1) val1 = 0.0f; else val1 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*k + (i+1)*dimY + (j)];
                if (j == dimY-1) val2 = 0.0f; else val2 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*k + (i)*dimY + (j+1)];
                if (k == dimZ-1) val3 = 0.0f; else val3 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*(k+1) + (i)*dimY + (j)];
                P1[(dimX*dimY)*k + (i)*dimY + (j)] = R1[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val1;
                P2[(dimX*dimY)*k + (i)*dimY + (j)] = R2[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val2;
                P3[(dimX*dimY)*k + (i)*dimY + (j)] = R3[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val3;
            }}}
    return 1;
}
float Proj_func3D(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{
    float val1, val2, val3;
    int i,j,k;
#pragma omp parallel for shared(P1,P2,P3) private(i,j,k,val1,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                val1 = fabs(P1[(dimX*dimY)*k + (i)*dimY + (j)]);
                val2 = fabs(P2[(dimX*dimY)*k + (i)*dimY + (j)]);
                val3 = fabs(P3[(dimX*dimY)*k + (i)*dimY + (j)]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                if (val3 < 1.0f) {val3 = 1.0f;}
                
                P1[(dimX*dimY)*k + (i)*dimY + (j)] = P1[(dimX*dimY)*k + (i)*dimY + (j)]/val1;
                P2[(dimX*dimY)*k + (i)*dimY + (j)] = P2[(dimX*dimY)*k + (i)*dimY + (j)]/val2;
                P3[(dimX*dimY)*k + (i)*dimY + (j)] = P3[(dimX*dimY)*k + (i)*dimY + (j)]/val3;
            }}}
    return 1;
}
float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, int dimX, int dimY, int dimZ)
{
    int i,j,k;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P3,P1_old,P2_old,P3_old,R1,R2,R3,multip) private(i,j,k)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                R1[(dimX*dimY)*k + (i)*dimY + (j)] = P1[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P1[(dimX*dimY)*k + (i)*dimY + (j)] - P1_old[(dimX*dimY)*k + (i)*dimY + (j)]);
                R2[(dimX*dimY)*k + (i)*dimY + (j)] = P2[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P2[(dimX*dimY)*k + (i)*dimY + (j)] - P2_old[(dimX*dimY)*k + (i)*dimY + (j)]);
                R3[(dimX*dimY)*k + (i)*dimY + (j)] = P3[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P3[(dimX*dimY)*k + (i)*dimY + (j)] - P3_old[(dimX*dimY)*k + (i)*dimY + (j)]);
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

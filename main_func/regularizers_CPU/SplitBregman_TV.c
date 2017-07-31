#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

/* C-OMP implementation of Split Bregman - TV denoising-regularization model (2D/3D)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon - tolerance constant [OPTIONAL parameter]
 * 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
 *
 * Output:
 * Filtered/regularized image
 *
 * Example:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
 * u = SplitBregman_TV(single(u0), 10, 30, 1e-04);
 *
 * to compile with OMP support: mex SplitBregman_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * References:
 * The Split Bregman Method for L1 Regularized Problems, by Tom Goldstein and Stanley Osher.
 * D. Kazantsev, 2016*
 */

float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);
float gauss_seidel2D(float *U,  float *A, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda, float mu);
float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda);
float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda);
float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY);

float gauss_seidel3D(float *U, float *A, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda, float mu);
float updDxDyDz_shrinkAniso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda);
float updDxDyDz_shrinkIso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda);
float updBxByBz3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ);

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, methTV;
    const int  *dim_array;
    float *A, *U=NULL, *U_old=NULL, *Dx=NULL, *Dy=NULL, *Dz=NULL, *Bx=NULL, *By=NULL, *Bz=NULL, lambda, mu, epsil, re, re1, re_old;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /*noisy image (2D/3D) */
    mu =  (float) mxGetScalar(prhs[1]); /* regularization parameter */
    iter = 35; /* default iterations number */
    epsil = 0.0001; /* default tolerance constant */
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
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input image must be in a single precision"); }
    
    lambda = 2.0f*mu;
    count = 1;
    re_old = 0.0f;
    /*Handling Matlab output data*/
    dimY = dim_array[0]; dimX = dim_array[1]; dimZ = dim_array[2];
    
    if (number_of_dims == 2) {
        dimZ = 1; /*2D case*/
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Dx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Dy = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        Bx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        By = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        
        copyIm(A, U, dimX, dimY, dimZ); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /*storing old values*/
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*GS iteration */
            gauss_seidel2D(U, A, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);
            
            if (methTV == 1)  updDxDy_shrinkAniso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
            else updDxDy_shrinkIso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
            
            updBxBy2D(U, Dx, Dy, Bx, By, dimX, dimY);
            
            /* calculate norm to terminate earlier */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(U_old[j] - U[j],2);
                re1 += pow(U_old[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
            if (count > 4) break;
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) break;
            }
            re_old = re;
            /*printf("%f %i %i \n", re, ll, count); */
            
            /*copyIm(U_old, U, dimX, dimY, dimZ); */
        }
        printf("SB iterations stopped at iteration: %i\n", ll);
    }
    if (number_of_dims == 3) {
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dy = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Dz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Bx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        By = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        Bz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        
        copyIm(A, U, dimX, dimY, dimZ); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /*storing old values*/
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*GS iteration */
            gauss_seidel3D(U, A, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda, mu);
            
            if (methTV == 1) updDxDyDz_shrinkAniso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            else updDxDyDz_shrinkIso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            
            updBxByBz3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ);
            
            /* calculate norm to terminate earlier */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<dimX*dimY*dimZ; j++)
            {
                re += pow(U[j] - U_old[j],2);
                re1 += pow(U[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
            if (count > 4) break;
            
            /* check that the residual norm is decreasing */
            if (ll > 2) {
                if (re > re_old) break; }
            /*printf("%f %i %i \n", re, ll, count); */
            re_old = re;
        }
        printf("SB iterations stopped at iteration: %i\n", ll);
    }
}

/* 2D-case related Functions */
/*****************************************************************/
float gauss_seidel2D(float *U, float *A, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda, float mu)
{
    float sum, normConst;
    int i,j,i1,i2,j1,j2;
    normConst = 1.0f/(mu + 4.0f*lambda);
    
#pragma omp parallel for shared(U) private(i,j,i1,i2,j1,j2,sum)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            
            sum = Dx[(i2)*dimY + (j)] - Dx[(i)*dimY + (j)] + Dy[(i)*dimY + (j2)] - Dy[(i)*dimY + (j)] - Bx[(i2)*dimY + (j)] + Bx[(i)*dimY + (j)] - By[(i)*dimY + (j2)] + By[(i)*dimY + (j)];
            sum += (U[(i1)*dimY + (j)] + U[(i2)*dimY + (j)] + U[(i)*dimY + (j1)] + U[(i)*dimY + (j2)]);
            sum *= lambda;
            sum += mu*A[(i)*dimY + (j)];
            U[(i)*dimY + (j)] = normConst*sum;
        }}
    return *U;
}

float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1;
    float val1, val11, val2, val22, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(i,j,i1,j1,val1,val11,val2,val22)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            val1 = (U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) + Bx[(i)*dimY + (j)];
            val2 = (U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) + By[(i)*dimY + (j)];
            
            val11 = fabs(val1) - denom_lam; if (val11 < 0) val11 = 0;
            val22 = fabs(val2) - denom_lam; if (val22 < 0) val22 = 0;
            
            if (val1 !=0) Dx[(i)*dimY + (j)] = (val1/fabs(val1))*val11; else Dx[(i)*dimY + (j)] = 0;
            if (val2 !=0) Dy[(i)*dimY + (j)] = (val2/fabs(val2))*val22; else Dy[(i)*dimY + (j)] = 0;
            
        }}
    return 1;
}
float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1;
    float val1, val11, val2, denom, denom_lam;
    denom_lam = 1.0f/lambda;
    
#pragma omp parallel for shared(U,denom_lam) private(i,j,i1,j1,val1,val11,val2,denom)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            val1 = (U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) + Bx[(i)*dimY + (j)];
            val2 = (U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) + By[(i)*dimY + (j)];
            
            denom = sqrt(val1*val1 + val2*val2);
            
            val11 = (denom - denom_lam); if (val11 < 0) val11 = 0.0f;
            
            if (denom != 0.0f) {
                Dx[(i)*dimY + (j)] = val11*(val1/denom);
                Dy[(i)*dimY + (j)] = val11*(val2/denom);
            }
            else {
                Dx[(i)*dimY + (j)] = 0;
                Dy[(i)*dimY + (j)] = 0;
            }
        }}
    return 1;
}
float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY)
{
    int i,j,i1,j1;
#pragma omp parallel for shared(U) private(i,j,i1,j1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            
            Bx[(i)*dimY + (j)] = Bx[(i)*dimY + (j)] + ((U[(i1)*dimY + (j)] - U[(i)*dimY + (j)]) - Dx[(i)*dimY + (j)]);
            By[(i)*dimY + (j)] = By[(i)*dimY + (j)] + ((U[(i)*dimY + (j1)] - U[(i)*dimY + (j)]) - Dy[(i)*dimY + (j)]);
        }}
    return 1;
}


/* 3D-case related Functions */
/*****************************************************************/
float gauss_seidel3D(float *U, float *A, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda, float mu)
{
    float normConst, d_val, b_val, sum;
    int i,j,i1,i2,j1,j2,k,k1,k2;
    normConst = 1.0f/(mu + 6.0f*lambda);
#pragma omp parallel for shared(U) private(i,j,i1,i2,j1,j2,k,k1,k2,d_val,b_val,sum)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                i2 = i-1; if (i2 < 0) i2 = i+1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                j2 = j-1; if (j2 < 0) j2 = j+1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                k2 = k-1; if (k2 < 0) k2 = k+1;
                
                d_val = Dx[(dimX*dimY)*k + (i2)*dimY + (j)] - Dx[(dimX*dimY)*k + (i)*dimY + (j)] + Dy[(dimX*dimY)*k + (i)*dimY + (j2)] - Dy[(dimX*dimY)*k + (i)*dimY + (j)] + Dz[(dimX*dimY)*k2 + (i)*dimY + (j)] - Dz[(dimX*dimY)*k + (i)*dimY + (j)];
                b_val = -Bx[(dimX*dimY)*k + (i2)*dimY + (j)] + Bx[(dimX*dimY)*k + (i)*dimY + (j)] - By[(dimX*dimY)*k + (i)*dimY + (j2)] + By[(dimX*dimY)*k + (i)*dimY + (j)] - Bz[(dimX*dimY)*k2 + (i)*dimY + (j)] + Bz[(dimX*dimY)*k + (i)*dimY + (j)];
                sum =  d_val + b_val;
                sum += U[(dimX*dimY)*k + (i1)*dimY + (j)] + U[(dimX*dimY)*k + (i2)*dimY + (j)] + U[(dimX*dimY)*k + (i)*dimY + (j1)] + U[(dimX*dimY)*k + (i)*dimY + (j2)] + U[(dimX*dimY)*k1 + (i)*dimY + (j)] + U[(dimX*dimY)*k2 + (i)*dimY + (j)];
                sum *= lambda;
                sum += mu*A[(dimX*dimY)*k + (i)*dimY + (j)];
                U[(dimX*dimY)*k + (i)*dimY + (j)] = normConst*sum;
            }}}
    return *U;
}

float updDxDyDz_shrinkAniso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda)
{
    int i,j,i1,j1,k,k1,index;
    float val1, val11, val2, val22, val3, val33, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(index,i,j,i1,j1,k,k1,val1,val11,val2,val22,val3,val33)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                index = (dimX*dimY)*k + (i)*dimY + (j);
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[index]) + Bz[index];
                
                val11 = fabs(val1) - denom_lam; if (val11 < 0) val11 = 0;
                val22 = fabs(val2) - denom_lam; if (val22 < 0) val22 = 0;
                val33 = fabs(val3) - denom_lam; if (val33 < 0) val33 = 0;
                
                if (val1 !=0) Dx[index] = (val1/fabs(val1))*val11; else Dx[index] = 0;
                if (val2 !=0) Dy[index] = (val2/fabs(val2))*val22; else Dy[index] = 0;
                if (val3 !=0) Dz[index] = (val3/fabs(val3))*val33; else Dz[index] = 0;
                
            }}}
    return 1;
}
float updDxDyDz_shrinkIso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda)
{
    int i,j,i1,j1,k,k1,index;
    float val1, val11, val2, val3, denom, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(index,denom,i,j,i1,j1,k,k1,val1,val11,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                index = (dimX*dimY)*k + (i)*dimY + (j);
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[index]) + Bz[index];
                
                denom = sqrt(val1*val1 + val2*val2 + val3*val3);
                
                val11 = (denom - denom_lam); if (val11 < 0) val11 = 0.0f;
                
                if (denom != 0.0f) {
                    Dx[index] = val11*(val1/denom);
                    Dy[index] = val11*(val2/denom);
                    Dz[index] = val11*(val3/denom);
                }
                else {
                    Dx[index] = 0;
                    Dy[index] = 0;
                    Dz[index] = 0;
                }               
            }}}
    return 1;
}
float updBxByBz3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ)
{
    int i,j,k,i1,j1,k1;
#pragma omp parallel for shared(U) private(i,j,k,i1,j1,k1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                Bx[(dimX*dimY)*k + (i)*dimY + (j)] = Bx[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k + (i1)*dimY + (j)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dx[(dimX*dimY)*k + (i)*dimY + (j)]);
                By[(dimX*dimY)*k + (i)*dimY + (j)] = By[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k + (i)*dimY + (j1)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dy[(dimX*dimY)*k + (i)*dimY + (j)]);
                Bz[(dimX*dimY)*k + (i)*dimY + (j)] = Bz[(dimX*dimY)*k + (i)*dimY + (j)] + ((U[(dimX*dimY)*k1 + (i)*dimY + (j)] - U[(dimX*dimY)*k + (i)*dimY + (j)]) - Dz[(dimX*dimY)*k + (i)*dimY + (j)]);
                
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
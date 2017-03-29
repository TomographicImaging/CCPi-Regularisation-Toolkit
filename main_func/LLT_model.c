#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

#define EPS 0.001

/* C-OMP implementation of Lysaker, Lundervold and Tai (LLT) model of higher order regularization penalty
 *
 * Input Parameters:
 * 1. U0 - origanal noise image/volume
 * 2. lambda - regularization parameter
 * 3. tau - time-step  for explicit scheme 
 * 4. iter - iterations number
 * 5. epsil  - tolerance constant (to terminate earlier) 
 * 6. switcher - default is 0, switch to (1) to restrictive smoothing in Z dimension (in test)
 *
 * Output:
 * Filtered/regularized image
 *
 * Example:
 * figure;
 * Im = double(imread('lena_gray_256.tif'))/255;  % loading image
 * u0 = Im + .03*randn(size(Im)); % adding noise
 * [Den] = LLT_model(single(u0), 10, 0.1, 1);
 *
 *
 * to compile with OMP support: mex LLT_model.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 * References: Lysaker, Lundervold and Tai (LLT) 2003, IEEE
 *
 * 28.11.16/Harwell
 */
/* 2D functions */
float der2D(float *U, float *D1, float *D2, int dimX, int dimY, int dimZ);
float div_upd2D(float *U0, float *U, float *D1, float *D2, int dimX, int dimY, int dimZ, float lambda, float tau);

float der3D(float *U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ);
float div_upd3D(float *U0, float *U, float *D1, float *D2, float *D3,  unsigned short *Map, int switcher, int dimX, int dimY, int dimZ, float lambda, float tau);

float calcMap(float *U, unsigned short *Map, int dimX, int dimY, int dimZ);
float cleanMap(unsigned short *Map, int dimX, int dimY, int dimZ);

float copyIm(float *A, float *U, int dimX, int dimY, int dimZ);

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
        
{
    int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, switcher;
    const int  *dim_array;
    float *U0, *U=NULL, *U_old=NULL, *D1=NULL, *D2=NULL, *D3=NULL, lambda, tau, re, re1, epsil, re_old;
    unsigned short *Map=NULL;
    
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    U0  = (float *) mxGetData(prhs[0]); /*origanal noise image/volume*/
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {mexErrMsgTxt("The input in single precision is required"); }
    lambda =  (float) mxGetScalar(prhs[1]); /*regularization parameter*/
    tau =  (float) mxGetScalar(prhs[2]); /* time-step */
    iter =  (int) mxGetScalar(prhs[3]); /*iterations number*/
    epsil =  (float) mxGetScalar(prhs[4]); /* tolerance constant */
    switcher =  (int) mxGetScalar(prhs[5]); /*switch on (1) restrictive smoothing in Z dimension*/
     
    /*Handling Matlab output data*/
    dimX = dim_array[0]; dimY = dim_array[1];  dimZ = 1;
    
    if (number_of_dims == 2) {
        /*2D case*/       
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
    }
    else if (number_of_dims == 3) {
        /*3D case*/
        dimZ = dim_array[2];
        U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        D3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
        if (switcher != 0) {
        Map = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array, mxUINT16_CLASS, mxREAL));       
        }
    }
    else {mexErrMsgTxt("The input data should be 2D or 3D");}
    
    /*Copy U0 to U*/
    copyIm(U0, U, dimX, dimY, dimZ);
    
    count = 1;
    re_old = 0.0f; 
    if (number_of_dims == 2) {
        for(ll = 0; ll < iter; ll++) {
            
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*estimate inner derrivatives */
            der2D(U, D1, D2, dimX, dimY, dimZ);
            /* calculate div^2 and update */
            div_upd2D(U0, U, D1, D2, dimX, dimY, dimZ, lambda, tau);
            
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
        
        } /*end of iterations*/
        printf("HO iterations stopped at iteration: %i\n", ll);          
    }
    /*3D version*/
    if (number_of_dims == 3) {
        
        if (switcher == 1) {
            /* apply restrictive smoothing */            
            calcMap(U, Map, dimX, dimY, dimZ);
            /*clear outliers */
            cleanMap(Map, dimX, dimY, dimZ);           
        }
        for(ll = 0; ll < iter; ll++) {
            
            copyIm(U, U_old, dimX, dimY, dimZ);
            
            /*estimate inner derrivatives */
            der3D(U, D1, D2, D3, dimX, dimY, dimZ);          
            /* calculate div^2 and update */
            div_upd3D(U0, U, D1, D2, D3, Map, switcher, dimX, dimY, dimZ, lambda, tau);                 
            
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
            
        } /*end of iterations*/
        printf("HO iterations stopped at iteration: %i\n", ll);       
    }
}

float der2D(float *U, float *D1, float *D2, int dimX, int dimY, int dimZ)
{
    int i, j, i_p, i_m, j_m, j_p;
    float dxx, dyy, denom_xx, denom_yy;
#pragma omp parallel for shared(U,D1,D2) private(i, j, i_p, i_m, j_m, j_p, denom_xx, denom_yy, dxx, dyy)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i_p = i + 1; if (i_p == dimX) i_p = i - 1;
            i_m = i - 1; if (i_m < 0) i_m = i + 1;
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            
            dxx = U[i_p*dimY + j] - 2.0f*U[i*dimY + j] + U[i_m*dimY + j];
            dyy = U[i*dimY + j_p] - 2.0f*U[i*dimY + j] + U[i*dimY + j_m];
                        
            denom_xx = fabs(dxx) + EPS;
            denom_yy = fabs(dyy) + EPS;
            
            D1[i*dimY + j] = dxx/denom_xx;
            D2[i*dimY + j] = dyy/denom_yy;
        }}
    return 1;
}
float div_upd2D(float *U0, float *U, float *D1, float *D2, int dimX, int dimY, int dimZ, float lambda, float tau)
{
    int i, j, i_p, i_m, j_m, j_p;
    float div, dxx, dyy;
#pragma omp parallel for shared(U,U0,D1,D2) private(i, j, i_p, i_m, j_m, j_p, div, dxx, dyy)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i_p = i + 1; if (i_p == dimX) i_p = i - 1;
            i_m = i - 1; if (i_m < 0) i_m = i + 1;
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            
            dxx = D1[i_p*dimY + j] - 2.0f*D1[i*dimY + j] + D1[i_m*dimY + j];
            dyy = D2[i*dimY + j_p] - 2.0f*D2[i*dimY + j] + D2[i*dimY + j_m];
            
            div = dxx + dyy;
            
            U[i*dimY + j] = U[i*dimY + j] - tau*div - tau*lambda*(U[i*dimY + j] - U0[i*dimY + j]);
        }}
    return *U0;
}

float der3D(float *U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ)
{
    int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m;
    float dxx, dyy, dzz, denom_xx, denom_yy, denom_zz;
#pragma omp parallel for shared(U,D1,D2,D3) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, denom_xx, denom_yy, denom_zz, dxx, dyy, dzz)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i_p = i + 1; if (i_p == dimX) i_p = i - 1;
        i_m = i - 1; if (i_m < 0) i_m = i + 1;
        for(j=0; j<dimY; j++) {
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            for(k=0; k<dimZ; k++) {
                k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
                k_m = k - 1; if (k_m < 0) k_m = k + 1;
                
                dxx = U[dimX*dimY*k + i_p*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i_m*dimY + j];
                dyy = U[dimX*dimY*k + i*dimY + j_p] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i*dimY + j_m];
                dzz = U[dimX*dimY*k_p + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m + i*dimY + j];                
                
                denom_xx = fabs(dxx) + EPS;
                denom_yy = fabs(dyy) + EPS;
                denom_zz = fabs(dzz) + EPS;
                
                D1[dimX*dimY*k + i*dimY + j] = dxx/denom_xx;
                D2[dimX*dimY*k + i*dimY + j] = dyy/denom_yy;
                D3[dimX*dimY*k + i*dimY + j] = dzz/denom_zz;               
                
            }}}
    return 1;
}

float div_upd3D(float *U0, float *U, float *D1, float *D2, float *D3,  unsigned short *Map, int switcher, int dimX, int dimY, int dimZ, float lambda, float tau)
{
    int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m;
    float div, dxx, dyy, dzz;
#pragma omp parallel for shared(U,U0,D1,D2,D3) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, div, dxx, dyy, dzz)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i_p = i + 1; if (i_p == dimX) i_p = i - 1;
        i_m = i - 1; if (i_m < 0) i_m = i + 1;
        for(j=0; j<dimY; j++) {
            j_p = j + 1; if (j_p == dimY) j_p = j - 1;
            j_m = j - 1; if (j_m < 0) j_m = j + 1;
            for(k=0; k<dimZ; k++) {
                k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
                k_m = k - 1; if (k_m < 0) k_m = k + 1;
//                 k_p1 = k + 2; if (k_p1 >= dimZ) k_p1 = k - 2;
//                 k_m1 = k - 2; if (k_m1 < 0) k_m1 = k + 2;   
                
                dxx = D1[dimX*dimY*k + i_p*dimY + j] - 2.0f*D1[dimX*dimY*k + i*dimY + j] + D1[dimX*dimY*k + i_m*dimY + j];
                dyy = D2[dimX*dimY*k + i*dimY + j_p] - 2.0f*D2[dimX*dimY*k + i*dimY + j] + D2[dimX*dimY*k + i*dimY + j_m];               
                dzz = D3[dimX*dimY*k_p + i*dimY + j] - 2.0f*D3[dimX*dimY*k + i*dimY + j] + D3[dimX*dimY*k_m + i*dimY + j];
                
                if ((switcher == 1) && (Map[dimX*dimY*k + i*dimY + j] == 0)) dzz = 0;                
                div = dxx + dyy + dzz;
                
//                 if (switcher == 1) {                    
                    // if (Map2[dimX*dimY*k + i*dimY + j] == 0) dzz2 = 0;
                    //else dzz2 = D4[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*D4[dimX*dimY*k + i*dimY + j] + D4[dimX*dimY*k_m1 + i*dimY + j];
//                     div = dzz + dzz2;
//                 }
                  
//                 dzz = D3[dimX*dimY*k_p + i*dimY + j] - 2.0f*D3[dimX*dimY*k + i*dimY + j] + D3[dimX*dimY*k_m + i*dimY + j];
//                 dzz2 = D4[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*D4[dimX*dimY*k + i*dimY + j] + D4[dimX*dimY*k_m1 + i*dimY + j];  
//                 div = dzz + dzz2;
                                
                U[dimX*dimY*k + i*dimY + j] = U[dimX*dimY*k + i*dimY + j] - tau*div - tau*lambda*(U[dimX*dimY*k + i*dimY + j] - U0[dimX*dimY*k + i*dimY + j]);
            }}}
        return *U0;
 }   

// float der3D_2(float *U, float *D1, float *D2, float *D3, float *D4, int dimX, int dimY, int dimZ)
// {
//     int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, k_p1, k_m1;
//     float dxx, dyy, dzz, dzz2, denom_xx, denom_yy, denom_zz, denom_zz2;
// #pragma omp parallel for shared(U,D1,D2,D3,D4) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, denom_xx, denom_yy, denom_zz, denom_zz2, dxx, dyy, dzz, dzz2, k_p1, k_m1)
//     for(i=0; i<dimX; i++) {
//         /* symmetric boundary conditions (Neuman) */
//         i_p = i + 1; if (i_p == dimX) i_p = i - 1;
//         i_m = i - 1; if (i_m < 0) i_m = i + 1;
//         for(j=0; j<dimY; j++) {
//             j_p = j + 1; if (j_p == dimY) j_p = j - 1;
//             j_m = j - 1; if (j_m < 0) j_m = j + 1;
//             for(k=0; k<dimZ; k++) {
//                 k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
//                 k_m = k - 1; if (k_m < 0) k_m = k + 1;
//                 k_p1 = k + 2; if (k_p1 >= dimZ) k_p1 = k - 2;
//                 k_m1 = k - 2; if (k_m1 < 0) k_m1 = k + 2;                
//                 
//                 dxx = U[dimX*dimY*k + i_p*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i_m*dimY + j];
//                 dyy = U[dimX*dimY*k + i*dimY + j_p] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i*dimY + j_m];
//                 dzz = U[dimX*dimY*k_p + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m + i*dimY + j];                
//                 dzz2 = U[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m1 + i*dimY + j];                
//                 
//                 denom_xx = fabs(dxx) + EPS;
//                 denom_yy = fabs(dyy) + EPS;
//                 denom_zz = fabs(dzz) + EPS;
//                 denom_zz2 = fabs(dzz2) + EPS;
//                 
//                 D1[dimX*dimY*k + i*dimY + j] = dxx/denom_xx;
//                 D2[dimX*dimY*k + i*dimY + j] = dyy/denom_yy;
//                 D3[dimX*dimY*k + i*dimY + j] = dzz/denom_zz;               
//                 D4[dimX*dimY*k + i*dimY + j] = dzz2/denom_zz2;                               
//             }}}
//     return 1;
// }

float calcMap(float *U, unsigned short *Map,  int dimX, int dimY, int dimZ)
{  
    int i,j,k,i1,j1,i2,j2,windowSize;
    float val1, val2,thresh_val,maxval; 
    windowSize = 1;
    thresh_val = 0.0001; /*thresh_val = 0.0035;*/
    
    /* normalize volume first */
    maxval = 0.0f;
     for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {  
             for(k=0; k<dimZ; k++) {  
                if (U[dimX*dimY*k + i*dimY + j] > maxval) maxval = U[dimX*dimY*k + i*dimY + j];
             }}}
    
    if (maxval != 0.0f) {
     for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {  
             for(k=0; k<dimZ; k++) {  
               U[dimX*dimY*k + i*dimY + j] = U[dimX*dimY*k + i*dimY + j]/maxval;
             }}}
    }
    else {
       printf("%s \n", "Maximum value is zero!");       
       return 0;
    }
    
    #pragma omp parallel for shared(U,Map) private(i, j, k, i1, j1, i2, j2, val1, val2)
     for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {  
             for(k=0; k<dimZ; k++) {         
                
                Map[dimX*dimY*k + i*dimY + j] = 0; 
//                 Map2[dimX*dimY*k + i*dimY + j] = 0; 
                
                val1 = 0.0f; val2 = 0.0f; 
                for(i1=-windowSize; i1<=windowSize; i1++) {
                    for(j1=-windowSize; j1<=windowSize; j1++) {
                    i2 = i+i1;
                    j2 = j+j1;
                    
                    if ((i2 >= 0) && (i2 < dimX) && (j2 >= 0) && (j2 < dimY)) {
                      if (k == 0) {
                          val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);                        
//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);                                                  
                      }
                      else if (k == dimZ-1) {
                          val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2);                           
                      }
//                       else if (k == 1) {
//                           val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
//                           val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);  
//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);                           
//                       }
//                       else if (k == dimZ-2) {
//                           val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
//                           val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);      
//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2);                           
//                       }                      
                      else {
                          val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
                          val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);                           
//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2); 
//                           val4 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);  
                      }
                    }                    
                    }}
                
                 val1 = 0.111f*val1; val2 = 0.111f*val2;
//                  val3 = 0.111f*val3; val4 = 0.111f*val4;
                 if ((val1 <= thresh_val) && (val2 <= thresh_val)) Map[dimX*dimY*k + i*dimY + j] = 1;                        
//                  if ((val3 <= thresh_val) && (val4 <= thresh_val)) Map2[dimX*dimY*k + i*dimY + j] = 1;                        
             }}}
     return 1;
}

float cleanMap(unsigned short *Map, int dimX, int dimY, int dimZ)
{  
    int i, j, k, i1, j1, i2, j2, counter; 
     #pragma omp parallel for shared(Map) private(i, j, k, i1, j1, i2, j2, counter)
     for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {  
             for(k=0; k<dimZ; k++) {   
                    
                 counter=0;
                 for(i1=-3; i1<=3; i1++) {
                    for(j1=-3; j1<=3; j1++) {
                    i2 = i+i1;
                    j2 = j+j1;
                    if ((i2 >= 0) && (i2 < dimX) && (j2 >= 0) && (j2 < dimY)) {
                    if (Map[dimX*dimY*k + i2*dimY + j2] == 0) counter++;                                       
                    }
                    }}
                 if (counter < 24) Map[dimX*dimY*k + i*dimY + j] = 1;
             }}}
    return *Map;
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
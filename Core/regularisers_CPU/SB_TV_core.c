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

#include "SB_TV_core.h"

/* C-OMP implementation of Split Bregman - TV denoising-regularisation model (2D/3D) [1]
*
* Input Parameters:
* 1. Noisy image/volume
* 2. lambda - regularisation parameter
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon - tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
* 6. print information: 0 (off) or 1 (on)  [OPTIONAL parameter]
*
* Output:
* 1. Filtered/regularized image
*
* [1]. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.
*/
 
float SB_TV_CPU_main(float *Input, float *Output, float mu, int iter, float epsil, int methodTV, int printM, int dimX, int dimY, int dimZ)
{
	int ll, j, DimTotal;
	float re, re1, lambda;
    int count = 0;
    mu = 1.0f/mu;
	lambda = 2.0f*mu;

	if (dimZ <= 1) {
		/* 2D case */
		float *Output_prev=NULL, *Dx=NULL, *Dy=NULL, *Bx=NULL, *By=NULL;
		DimTotal = dimX*dimY;
		
		Output_prev = calloc(DimTotal, sizeof(float));
		Dx = calloc(DimTotal, sizeof(float));
		Dy = calloc(DimTotal, sizeof(float));
		Bx = calloc(DimTotal, sizeof(float));
		By = calloc(DimTotal, sizeof(float));
        
        copyIm(Input, Output, dimX, dimY, 1); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /* storing old estimate */
            copyIm(Output, Output_prev, dimX, dimY, 1);
            
            /* perform two GS iterations (normally 2 is enough for the convergence) */
            gauss_seidel2D(Output, Input, Output_prev, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);
            copyIm(Output, Output_prev, dimX, dimY, 1);
            /*GS iteration */
            gauss_seidel2D(Output, Input, Output_prev, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);
            
            /* TV-related step */
            if (methodTV == 1)  updDxDy_shrinkAniso2D(Output, Dx, Dy, Bx, By, dimX, dimY, lambda);
            else updDxDy_shrinkIso2D(Output, Dx, Dy, Bx, By, dimX, dimY, lambda);
            
            /* update for Bregman variables */
            updBxBy2D(Output, Dx, Dy, Bx, By, dimX, dimY);
            
            /* check early stopping criteria if epsilon not equal zero */
            if (epsil != 0) {
            re = 0.0f; re1 = 0.0f;
				for(j=0; j<DimTotal; j++) {
                re += pow(Output[j] - Output_prev[j],2);
                re1 += pow(Output[j],2);
				}
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
				if (count > 4) break;
			}
            /*printf("%f %i %i \n", re, ll, count); */
        }
        if (printM == 1) printf("SB-TV iterations stopped at iteration %i \n", ll);
		free(Output_prev); free(Dx); free(Dy); free(Bx); free(By);
	}
	else {
		/* 3D case */
		float *Output_prev=NULL, *Dx=NULL, *Dy=NULL, *Dz=NULL, *Bx=NULL, *By=NULL, *Bz=NULL;
		DimTotal = dimX*dimY*dimZ;
		
		Output_prev = calloc(DimTotal, sizeof(float));
		Dx = calloc(DimTotal, sizeof(float));
		Dy = calloc(DimTotal, sizeof(float));
		Dz = calloc(DimTotal, sizeof(float));
		Bx = calloc(DimTotal, sizeof(float));
		By = calloc(DimTotal, sizeof(float));
		Bz = calloc(DimTotal, sizeof(float));
        
        copyIm(Input, Output, dimX, dimY, dimZ); /*initialize */
        
        /* begin outer SB iterations */
        for(ll=0; ll<iter; ll++) {
            
            /* storing old estimate */
            copyIm(Output, Output_prev, dimX, dimY, dimZ);
            
             /* perform two GS iterations (normally 2 is enough for the convergence) */
            gauss_seidel3D(Output, Input, Output_prev, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda, mu);
            copyIm(Output, Output_prev, dimX, dimY, dimZ);
            /*GS iteration */
            gauss_seidel3D(Output, Input, Output_prev, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda, mu);
            
            /* TV-related step */
            if (methodTV == 1)  updDxDyDz_shrinkAniso3D(Output, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            else updDxDyDz_shrinkIso3D(Output, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
            
            /* update for Bregman variables */
            updBxByBz3D(Output, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ);
            
            /* check early stopping criteria if epsilon not equal zero */
            if (epsil != 0) {
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<DimTotal; j++) {
                re += pow(Output[j] - Output_prev[j],2);
                re1 += pow(Output[j],2);
				}
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
				if (count > 4) break;
			}
            /*printf("%f %i %i \n", re, ll, count); */
        }
        if (printM == 1) printf("SB-TV iterations stopped at iteration %i \n", ll);
		free(Output_prev); free(Dx); free(Dy); free(Dz); free(Bx); free(By); free(Bz);
	}
	return *Output;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float gauss_seidel2D(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda, float mu)
{
    float sum, normConst;
    int i,j,i1,i2,j1,j2,index;
    normConst = 1.0f/(mu + 4.0f*lambda);
    
#pragma omp parallel for shared(U) private(index,i,j,i1,i2,j1,j2,sum)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            index = j*dimX+i;
            
            sum = Dx[j*dimX+i2] - Dx[index] + Dy[j2*dimX+i] - Dy[index] - Bx[j*dimX+i2] + Bx[index] - By[j2*dimX+i] + By[index];
            sum += U_prev[j*dimX+i1] + U_prev[j*dimX+i2] + U_prev[j1*dimX+i] + U_prev[j2*dimX+i];
            sum *= lambda;
            sum += mu*A[index];
            U[index] = normConst*sum;
        }}
    return *U;
}

float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1,index;
    float val1, val11, val2, val22, denom_lam;
    denom_lam = 1.0f/lambda;
#pragma omp parallel for shared(U,denom_lam) private(index,i,j,i1,j1,val1,val11,val2,val22)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            index = j*dimX+i;
            
            val1 = (U[j*dimX+i1] - U[index]) + Bx[index];
            val2 = (U[j1*dimX+i] - U[index]) + By[index];
            
            val11 = fabs(val1) - denom_lam; if (val11 < 0) val11 = 0;
            val22 = fabs(val2) - denom_lam; if (val22 < 0) val22 = 0;
            
            if (val1 !=0) Dx[index] = (val1/fabs(val1))*val11; else Dx[index] = 0;
            if (val2 !=0) Dy[index] = (val2/fabs(val2))*val22; else Dy[index] = 0;
            
        }}
    return 1;
}
float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda)
{
    int i,j,i1,j1,index;
    float val1, val11, val2, denom, denom_lam;
    denom_lam = 1.0f/lambda;
    
#pragma omp parallel for shared(U,denom_lam) private(index,i,j,i1,j1,val1,val11,val2,denom)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            index = j*dimX+i;
            
            val1 = (U[j*dimX+i1] - U[index]) + Bx[index];
            val2 = (U[j1*dimX+i] - U[index]) + By[index];
            
            denom = sqrt(val1*val1 + val2*val2);
            
            val11 = (denom - denom_lam); if (val11 < 0) val11 = 0.0f;
            
            if (denom != 0.0f) {
                Dx[index] = val11*(val1/denom);
                Dy[index] = val11*(val2/denom);
            }
            else {
                Dx[index] = 0;
                Dy[index] = 0;
            }
        }}
    return 1;
}
float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY)
{
    int i,j,i1,j1,index;
#pragma omp parallel for shared(U) private(index,i,j,i1,j1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == dimX) i1 = i-1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            index = j*dimX+i;
            
            Bx[index] += (U[j*dimX+i1] - U[index]) - Dx[index];
            By[index] += (U[j1*dimX+i] - U[index]) - Dy[index];
        }}
    return 1;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/*****************************************************************/
float gauss_seidel3D(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda, float mu)
{
    float normConst, d_val, b_val, sum;
    int i,j,i1,i2,j1,j2,k,k1,k2,index;
    normConst = 1.0f/(mu + 6.0f*lambda);
#pragma omp parallel for shared(U) private(index,i,j,i1,i2,j1,j2,k,k1,k2,d_val,b_val,sum)
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
                index = (dimX*dimY)*k + j*dimX+i;
                
                d_val = Dx[(dimX*dimY)*k + j*dimX+i2] - Dx[index] + Dy[(dimX*dimY)*k + j2*dimX+i] - Dy[index] + Dz[(dimX*dimY)*k2 + j*dimX+i] - Dz[index];
                b_val = -Bx[(dimX*dimY)*k + j*dimX+i2] + Bx[index] - By[(dimX*dimY)*k + j2*dimX+i] + By[index] - Bz[(dimX*dimY)*k2 + j*dimX+i] + Bz[index];
                sum = d_val + b_val;
                sum += U_prev[(dimX*dimY)*k + j*dimX+i1] + U_prev[(dimX*dimY)*k + j*dimX+i2] + U_prev[(dimX*dimY)*k + j1*dimX+i] + U_prev[(dimX*dimY)*k + j2*dimX+i] + U_prev[(dimX*dimY)*k1 + j*dimX+i] + U_prev[(dimX*dimY)*k2 + j*dimX+i];
                sum *= lambda;
                sum += mu*A[index];
                U[index] = normConst*sum;
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
                index = (dimX*dimY)*k + j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + j*dimX+i1] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + j1*dimX+i] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + j*dimX+i] - U[index]) + Bz[index];
                
                val11 = fabs(val1) - denom_lam; if (val11 < 0.0f) val11 = 0.0f;
                val22 = fabs(val2) - denom_lam; if (val22 < 0.0f) val22 = 0.0f;
                val33 = fabs(val3) - denom_lam; if (val33 < 0.0f) val33 = 0.0f;
                
                if (val1 !=0.0f) Dx[index] = (val1/fabs(val1))*val11; else Dx[index] = 0.0f;
                if (val2 !=0.0f) Dy[index] = (val2/fabs(val2))*val22; else Dy[index] = 0.0f;
                if (val3 !=0.0f) Dz[index] = (val3/fabs(val3))*val33; else Dz[index] = 0.0f;
                
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
                index = (dimX*dimY)*k + j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                val1 = (U[(dimX*dimY)*k + j*dimX+i1] - U[index]) + Bx[index];
                val2 = (U[(dimX*dimY)*k + j1*dimX+i] - U[index]) + By[index];
                val3 = (U[(dimX*dimY)*k1 + j*dimX+i] - U[index]) + Bz[index];
                
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
    int i,j,k,i1,j1,k1,index;
#pragma omp parallel for shared(U) private(index,i,j,k,i1,j1,k1)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
				index = (dimX*dimY)*k + j*dimX+i;
                /* symmetric boundary conditions (Neuman) */
                i1 = i+1; if (i1 == dimX) i1 = i-1;
                j1 = j+1; if (j1 == dimY) j1 = j-1;
                k1 = k+1; if (k1 == dimZ) k1 = k-1;
                
                Bx[index] += (U[(dimX*dimY)*k + j*dimX+i1] - U[index]) - Dx[index];
                By[index] += (U[(dimX*dimY)*k + j1*dimX+i] - U[index]) - Dy[index];
                Bz[index] += (U[(dimX*dimY)*k1 + j*dimX+i] - U[index]) - Dz[index];
            }}}
    return 1;
}

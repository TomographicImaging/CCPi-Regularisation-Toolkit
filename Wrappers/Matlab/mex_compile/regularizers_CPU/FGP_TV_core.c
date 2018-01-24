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

#include "FGP_TV_core.h"

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
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 *
 * D. Kazantsev, 2016-17
 *
 */

/* 2D-case related Functions */
/*****************************************************************/
float Obj_func_CALC2D(float *A, float *D, float *funcvalA, float lambda, int dimX, int dimY)
{   
    int i,j;
    float f1, f2, val1, val2;
    
    /*data-related term */
    f1 = 0.0f;
    for(i=0; i<dimX*dimY; i++) f1 += pow(D[i] - A[i],2);    
    
    /*TV-related term */
    f2 = 0.0f;
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            /* boundary conditions  */
            if (i == dimX-1) {val1 = 0.0f;} else {val1 = A[(i+1)*dimY + (j)] - A[(i)*dimY + (j)];}
            if (j == dimY-1) {val2 = 0.0f;} else {val2 = A[(i)*dimY + (j+1)] - A[(i)*dimY + (j)];}    
            f2 += sqrt(pow(val1,2) + pow(val2,2));
        }}  
    
    /* sum of two terms */
    funcvalA[0] = 0.5f*f1 + lambda*f2;     
    return *funcvalA;
}

float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
	float val1, val2;
	int i, j;
#pragma omp parallel for shared(A,D,R1,R2) private(i,j,val1,val2)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* boundary conditions  */
			if (i == 0) { val1 = 0.0f; }
			else { val1 = R1[(i - 1)*dimY + (j)]; }
			if (j == 0) { val2 = 0.0f; }
			else { val2 = R2[(i)*dimY + (j - 1)]; }
			D[(i)*dimY + (j)] = A[(i)*dimY + (j)] - lambda*(R1[(i)*dimY + (j)] + R2[(i)*dimY + (j)] - val1 - val2);
		}
	}
	return *D;
}
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY)
{
	float val1, val2, multip;
	int i, j;
	multip = (1.0f / (8.0f*lambda));
#pragma omp parallel for shared(P1,P2,D,R1,R2,multip) private(i,j,val1,val2)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* boundary conditions */
			if (i == dimX - 1) val1 = 0.0f; else val1 = D[(i)*dimY + (j)] - D[(i + 1)*dimY + (j)];
			if (j == dimY - 1) val2 = 0.0f; else val2 = D[(i)*dimY + (j)] - D[(i)*dimY + (j + 1)];
			P1[(i)*dimY + (j)] = R1[(i)*dimY + (j)] + multip*val1;
			P2[(i)*dimY + (j)] = R2[(i)*dimY + (j)] + multip*val2;
		}
	}
	return 1;
}
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY)
{
	float val1, val2, denom;
	int i, j;
	if (methTV == 0) {
		/* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,denom)
		for (i = 0; i<dimX; i++) {
			for (j = 0; j<dimY; j++) {
				denom = pow(P1[(i)*dimY + (j)], 2) + pow(P2[(i)*dimY + (j)], 2);
				if (denom > 1) {
					P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] / sqrt(denom);
					P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] / sqrt(denom);
				}
			}
		}
	}
	else {
		/* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,j,val1,val2)
		for (i = 0; i<dimX; i++) {
			for (j = 0; j<dimY; j++) {
				val1 = fabs(P1[(i)*dimY + (j)]);
				val2 = fabs(P2[(i)*dimY + (j)]);
				if (val1 < 1.0f) { val1 = 1.0f; }
				if (val2 < 1.0f) { val2 = 1.0f; }
				P1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] / val1;
				P2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] / val2;
			}
		}
	}
	return 1;
}
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY)
{
	int i, j;
	float multip;
	multip = ((tk - 1.0f) / tkp1);
#pragma omp parallel for shared(P1,P2,P1_old,P2_old,R1,R2,multip) private(i,j)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			R1[(i)*dimY + (j)] = P1[(i)*dimY + (j)] + multip*(P1[(i)*dimY + (j)] - P1_old[(i)*dimY + (j)]);
			R2[(i)*dimY + (j)] = P2[(i)*dimY + (j)] + multip*(P2[(i)*dimY + (j)] - P2_old[(i)*dimY + (j)]);
		}
	}
	return 1;
}

/* 3D-case related Functions */
/*****************************************************************/
float Obj_func_CALC3D(float *A, float *D, float *funcvalA, float lambda, int dimX, int dimY, int dimZ)
{   
    int i,j,k;
    float f1, f2, val1, val2, val3;
    
    /*data-related term */
    f1 = 0.0f;
    for(i=0; i<dimX*dimY*dimZ; i++) f1 += pow(D[i] - A[i],2);    
    
    /*TV-related term */
    f2 = 0.0f;
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
            /* boundary conditions  */
            if (i == dimX-1) {val1 = 0.0f;} else {val1 = A[(dimX*dimY)*k + (i+1)*dimY + (j)] - A[(dimX*dimY)*k + (i)*dimY + (j)];}
            if (j == dimY-1) {val2 = 0.0f;} else {val2 = A[(dimX*dimY)*k + (i)*dimY + (j+1)] - A[(dimX*dimY)*k + (i)*dimY + (j)];}    
            if (k == dimZ-1) {val3 = 0.0f;} else {val3 = A[(dimX*dimY)*(k+1) + (i)*dimY + (j)] - A[(dimX*dimY)*k + (i)*dimY + (j)];}    
            f2 += sqrt(pow(val1,2) + pow(val2,2)  + pow(val3,2));
        }}}     
    /* sum of two terms */
    funcvalA[0] = 0.5f*f1 + lambda*f2;     
    return *funcvalA;
}

float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ)
{
	float val1, val2, val3;
	int i, j, k;
#pragma omp parallel for shared(A,D,R1,R2,R3) private(i,j,k,val1,val2,val3)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {
				/* boundary conditions */
				if (i == 0) { val1 = 0.0f; }
				else { val1 = R1[(dimX*dimY)*k + (i - 1)*dimY + (j)]; }
				if (j == 0) { val2 = 0.0f; }
				else { val2 = R2[(dimX*dimY)*k + (i)*dimY + (j - 1)]; }
				if (k == 0) { val3 = 0.0f; }
				else { val3 = R3[(dimX*dimY)*(k - 1) + (i)*dimY + (j)]; }
				D[(dimX*dimY)*k + (i)*dimY + (j)] = A[(dimX*dimY)*k + (i)*dimY + (j)] - lambda*(R1[(dimX*dimY)*k + (i)*dimY + (j)] + R2[(dimX*dimY)*k + (i)*dimY + (j)] + R3[(dimX*dimY)*k + (i)*dimY + (j)] - val1 - val2 - val3);
			}
		}
	}
	return *D;
}
float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ)
{
	float val1, val2, val3, multip;
	int i, j, k;
	multip = (1.0f / (8.0f*lambda));
#pragma omp parallel for shared(P1,P2,P3,D,R1,R2,R3,multip) private(i,j,k,val1,val2,val3)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {
				/* boundary conditions */
				if (i == dimX - 1) val1 = 0.0f; else val1 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*k + (i + 1)*dimY + (j)];
				if (j == dimY - 1) val2 = 0.0f; else val2 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*k + (i)*dimY + (j + 1)];
				if (k == dimZ - 1) val3 = 0.0f; else val3 = D[(dimX*dimY)*k + (i)*dimY + (j)] - D[(dimX*dimY)*(k + 1) + (i)*dimY + (j)];
				P1[(dimX*dimY)*k + (i)*dimY + (j)] = R1[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val1;
				P2[(dimX*dimY)*k + (i)*dimY + (j)] = R2[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val2;
				P3[(dimX*dimY)*k + (i)*dimY + (j)] = R3[(dimX*dimY)*k + (i)*dimY + (j)] + multip*val3;
			}
		}
	}
	return 1;
}
float Proj_func3D(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{
	float val1, val2, val3;
	int i, j, k;
#pragma omp parallel for shared(P1,P2,P3) private(i,j,k,val1,val2,val3)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {
				val1 = fabs(P1[(dimX*dimY)*k + (i)*dimY + (j)]);
				val2 = fabs(P2[(dimX*dimY)*k + (i)*dimY + (j)]);
				val3 = fabs(P3[(dimX*dimY)*k + (i)*dimY + (j)]);
				if (val1 < 1.0f) { val1 = 1.0f; }
				if (val2 < 1.0f) { val2 = 1.0f; }
				if (val3 < 1.0f) { val3 = 1.0f; }

				P1[(dimX*dimY)*k + (i)*dimY + (j)] = P1[(dimX*dimY)*k + (i)*dimY + (j)] / val1;
				P2[(dimX*dimY)*k + (i)*dimY + (j)] = P2[(dimX*dimY)*k + (i)*dimY + (j)] / val2;
				P3[(dimX*dimY)*k + (i)*dimY + (j)] = P3[(dimX*dimY)*k + (i)*dimY + (j)] / val3;
			}
		}
	}
	return 1;
}
float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, int dimX, int dimY, int dimZ)
{
	int i, j, k;
	float multip;
	multip = ((tk - 1.0f) / tkp1);
#pragma omp parallel for shared(P1,P2,P3,P1_old,P2_old,P3_old,R1,R2,R3,multip) private(i,j,k)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {
				R1[(dimX*dimY)*k + (i)*dimY + (j)] = P1[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P1[(dimX*dimY)*k + (i)*dimY + (j)] - P1_old[(dimX*dimY)*k + (i)*dimY + (j)]);
				R2[(dimX*dimY)*k + (i)*dimY + (j)] = P2[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P2[(dimX*dimY)*k + (i)*dimY + (j)] - P2_old[(dimX*dimY)*k + (i)*dimY + (j)]);
				R3[(dimX*dimY)*k + (i)*dimY + (j)] = P3[(dimX*dimY)*k + (i)*dimY + (j)] + multip*(P3[(dimX*dimY)*k + (i)*dimY + (j)] - P3_old[(dimX*dimY)*k + (i)*dimY + (j)]);
			}
		}
	}
	return 1;
}



/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazanteev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
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

#include "utils.h"
#include <math.h>

/* Copy Image (float) */
float copyIm(float *A, float *U, long dimX, long dimY, long dimZ)
{
    long j;
#pragma omp parallel for shared(A, U) private(j)
    for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
    return *U;
}

/* Copy Image */
unsigned char copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, U) private(j)
    for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
    return *U;
}

/*Roll image symmetrically from top to bottom*/
float copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher)
{
    int i, j;
#pragma omp parallel for shared(U, A) private(i,j)
    for (i=0; i<dimX; i++) {
        for (j=0; j<dimY; j++) {
            if (switcher == 0) {
                if (j < (dimY - roll_value)) U[j*dimX + i] = A[(j+roll_value)*dimX + i];
                else U[j*dimX + i] = A[(j - (dimY - roll_value))*dimX + i];
            }
            else {
                if (j < roll_value) U[j*dimX + i] = A[(j+(dimY - roll_value))*dimX + i];
                else U[j*dimX + i] = A[(j - roll_value)*dimX + i];
            }
        }}
    return *U;
}

/* function that calculates TV energy
 * type - 1:  2*lambda*min||\nabla u|| + ||u -u0||^2
 * type - 2:  2*lambda*min||\nabla u||
 * */
float TV_energy2D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

    int i, j, index, thread_id;
	float x_val, y_val, fid;

	float E_Grad = 0;
	float * E_Grad_local = new float[num_threads];
	memset(E_Grad_local, 0, num_threads * sizeof(float));

	float E_Data = 0;
	float * E_Data_local = new float[omp_get_num_threads()];
	memset(E_Data_local, 0, num_threads * sizeof(float));


#pragma omp parallel for private (i, j, index, x_val, y_val, fid, thread_id)
	for (j = 0; j < dimY - 1; j++)
	{
		index = j*dimX;
		thread_id = omp_get_thread_num();

		for (i = 0; i < dimX - 1; i++)
		{
			x_val = U[index] - U[index + 1];
			y_val = U[index] - U[index + dimX];
			fid = U[index] - U0[index];

			E_Grad_local[thread_id] += 2.0f*lambda*sqrtf(x_val * x_val + y_val * y_val);
			E_Data_local[thread_id] += fid * fid;

			index++;
		}
	}

	for (int i = 0; i < num_threads; i++)
	{
		E_Grad += E_Grad_local[i];
		E_Data += E_Data_local[i];
	}

    if (type == 1) E_val[0] = E_Grad + E_Data;
    if (type == 2) E_val[0] = E_Grad;
}
float TV_energy3D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY, int dimZ)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

	int i, j, k, index, thread_id;
	float x_val, y_val, z_val, fid;

	float E_Grad = 0;
	float * E_Grad_local = new float[num_threads];
	memset(E_Grad_local, 0, num_threads * sizeof(float));

	float E_Data = 0;
	float * E_Data_local = new float[omp_get_num_threads()];
	memset(E_Data_local, 0, num_threads * sizeof(float));


	for (k = 0; k < dimZ - 1; k++)
	{
#pragma omp parallel for private (i, j, k, index, x_val, y_val, z_val, fid, thread_id)
		for (j = 0; j < dimY - 1; j++)
		{
			thread_id = omp_get_thread_num();
			index = k*dimX*dimY + j*dimX;

			for (i = 0; i < dimX - 1; i++)
			{
				x_val = U[index] - U[index + 1];
				y_val = U[index] - U[index + dimX];
				fid = U[index] - U0[index];

				E_Grad_local[thread_id] += 2.0f*lambda*sqrtf(x_val * x_val + y_val * y_val);
				E_Data_local[thread_id] += fid * fid;

				index++;
			}
		}
	}

	for (int i = 0; i < num_threads; i++)
	{
		E_Grad += E_Grad_local[i];
		E_Data += E_Data_local[i];
	}

	if (type == 1) E_val[0] = E_Grad + E_Data;
	if (type == 2) E_val[0] = E_Grad;
}

/* Down-Up scaling of 2D images using bilinear interpolation */
float Im_scale2D(float *Input, float *Scaled, int w, int h, int w2, int h2)
{
    int x, y, index, i, j;
    float x_ratio = ((float)(w-1))/w2;
    float y_ratio = ((float)(h-1))/h2;
    float A, B, C, D, x_diff, y_diff, gray;
    #pragma omp parallel for shared (Input, Scaled) private(x, y, index, A, B, C, D, x_diff, y_diff, gray)
    for (j=0;j<w2;j++) {
        for (i=0;i<h2;i++) {
            x = (int)(x_ratio * j);
            y = (int)(y_ratio * i);
            x_diff = (x_ratio * j) - x;
            y_diff = (y_ratio * i) - y;
            index = y*w+x ;

            A = Input[index];
            B = Input[index+1];
            C = Input[index+w];
            D = Input[index+w+1];

            gray = (float)(A*(1.0 - x_diff)*(1.0 - y_diff) +  B*(x_diff)*(1.0 - y_diff) +
                    C*(y_diff)*(1.0 - x_diff) +  D*(x_diff*y_diff));

            Scaled[i*w2+j] = gray;
        }}
    return *Scaled;
}

/*2D Projection onto convex set for P (called in PD_TV, FGP_dTV and FGP_TV methods)*/
float Proj_func2D(float *P1, float *P2, int methTV, long DimTotal)
{
    float val1, val2, denom, sq_denom;
    long i;
    if (methTV == 0) {
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,denom,sq_denom)
        for(i=0; i<DimTotal; i++) {
            denom = powf(P1[i],2) +  powf(P2[i],2);
            if (denom > 1.0f) {
                sq_denom = 1.0f/sqrtf(denom);
                P1[i] = P1[i]*sq_denom;
                P2[i] = P2[i]*sq_denom;
            }
        }
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,val1,val2)
        for(i=0; i<DimTotal; i++) {
            val1 = fabs(P1[i]);
            val2 = fabs(P2[i]);
            if (val1 < 1.0f) {val1 = 1.0f;}
            if (val2 < 1.0f) {val2 = 1.0f;}
            P1[i] = P1[i]/val1;
            P2[i] = P2[i]/val2;
        }
    }
    return 1;
}
/*3D Projection onto convex set for P (called in PD_TV, FGP_TV, FGP_dTV methods)*/
float Proj_func3D(float *P1, float *P2, float *P3, int methTV, long DimTotal)
{
    float val1, val2, val3, denom, sq_denom;
    long i;
    if (methTV == 0) {
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3,sq_denom)
        for(i=0; i<DimTotal; i++) {
            denom = powf(P1[i],2) + powf(P2[i],2) + powf(P3[i],2);
            if (denom > 1.0f) {
                sq_denom = 1.0f/sqrtf(denom);
                P1[i] = P1[i]*sq_denom;
                P2[i] = P2[i]*sq_denom;
                P3[i] = P3[i]*sq_denom;
            }
        }
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3)
        for(i=0; i<DimTotal; i++) {
            val1 = fabs(P1[i]);
            val2 = fabs(P2[i]);
            val3 = fabs(P3[i]);
            if (val1 < 1.0f) {val1 = 1.0f;}
            if (val2 < 1.0f) {val2 = 1.0f;}
            if (val3 < 1.0f) {val3 = 1.0f;}
            P1[i] = P1[i]/val1;
            P2[i] = P2[i]/val2;
            P3[i] = P3[i]/val3;
        }
    }
    return 1;
}

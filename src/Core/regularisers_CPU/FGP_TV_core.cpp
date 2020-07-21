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

#include "FGP_TV_core.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default)
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 */

 /* pointer swap*/
static inline void swap(float*&m, float*&n)
{
	float* tmp = m;
	m = n;
	n = tmp;
}

float TV_FGP_CPU(const float *Input, float *Output, float *infovector, float lambdaPar, int iterationsNumb, float epsil, int methodTV, int nonneg, int dimX, int dimY, int dimZ)
{
    int ll;
    long DimTotal;
    float re;
    re = 0.0f;
    float tk = 1.0f;
    float tkp1 =1.0f;
    int count = 0;

    if (dimZ <= 1) {
        /*2D case */
        float *Output_prev=NULL, *P1=NULL, *P2=NULL, *P1_prev=NULL, *P2_prev=NULL, *R1=NULL, *R2=NULL;
        DimTotal = (long)dimX* (long)dimY;

        if (epsil != 0.0f) Output_prev = (float*)malloc(DimTotal*sizeof(float));

        P1 = (float*)calloc(DimTotal, sizeof(float));
        P2 = (float*)calloc(DimTotal, sizeof(float));
        P1_prev = (float*)calloc(DimTotal, sizeof(float));
        P2_prev = (float*)calloc(DimTotal, sizeof(float));
        R1 = (float*)calloc(DimTotal, sizeof(float));
        R2 = (float*)calloc(DimTotal, sizeof(float));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

			if ((epsil != 0.0f) && (ll % 5 == 0))
				swap(Output, Output_prev);

            /* computing the gradient of the objective function */
            Obj_func(Input, Output, R1, R2, lambdaPar, (long)(dimX), (long)(dimY));

            /* apply nonnegativity */
			if (nonneg == 1) apply_nonnegativity(Output, DimTotal);

            /*Taking a step towards minus of the gradient*/
            Grad_func(P1, P2, Output, R1, R2, lambdaPar, (long)(dimX), (long)(dimY));

            /* projection step */
            Proj_func(P1, P2, methodTV, DimTotal);

            /*updating R and t*/
            tkp1 = (1.0f + sqrtf(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func(P1, P1_prev, P2, P2_prev, R1, R2, tkp1, tk, DimTotal);

            /* check early stopping criteria */
            if ((epsil != 0.0f)  && (ll % 5 == 0))
			{
				calculate_norm(Output, Output_prev, &re, DimTotal);

                if (re < epsil)  count++;
                if (count > 3) break;
            }

			swap(P1, P1_prev);
			swap(P2, P2_prev);

			tk = tkp1;
        }

        if (epsil != 0.0f) free(Output_prev);
        free(P1); free(P2); free(P1_prev); free(P2_prev); free(R1); free(R2);
    }
    else {
        /*3D case*/
        float *Output_prev=NULL, *P1=NULL, *P2=NULL, *P3=NULL, *P1_prev=NULL, *P2_prev=NULL, *P3_prev=NULL, *R1=NULL, *R2=NULL, *R3=NULL;
		DimTotal = (long)dimX * (long)dimY * (long)dimZ;

        if (epsil != 0.0f) Output_prev = (float*)calloc(DimTotal, sizeof(float));
        P1 = (float*)calloc(DimTotal, sizeof(float));
        P2 = (float*)calloc(DimTotal, sizeof(float));
        P3 = (float*)calloc(DimTotal, sizeof(float));
        P1_prev = (float*)calloc(DimTotal, sizeof(float));
        P2_prev = (float*)calloc(DimTotal, sizeof(float));
        P3_prev = (float*)calloc(DimTotal, sizeof(float));
        R1 = (float*)calloc(DimTotal, sizeof(float));
        R2 = (float*)calloc(DimTotal, sizeof(float));
        R3 = (float*)calloc(DimTotal, sizeof(float));

        /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {

            if ((epsil != 0.0f)  && (ll % 5 == 0))
				swap(Output, Output_prev);

            /* computing the gradient of the objective function */
            Obj_func(Input, Output, R1, R2, R3, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));

            /* apply nonnegativity */
			if (nonneg == 1) apply_nonnegativity(Output, DimTotal);

            /*Taking a step towards minus of the gradient*/
            Grad_func(P1, P2, P3, Output, R1, R2, R3, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));

            /* projection step */
            Proj_func(P1, P2, P3, methodTV, DimTotal);

            /*updating R and t*/
            tkp1 = (1.0f + sqrtf(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func(P1, P1_prev, P2, P2_prev, P3, P3_prev, R1, R2, R3, tkp1, tk, DimTotal);

            /* calculate norm - stopping rules*/
            if ((epsil != 0.0f)  && (ll % 5 == 0)) 
			{
				calculate_norm(Output, Output_prev, &re, DimTotal);

				/* stop if the norm residual is less than the tolerance EPS */
				if (re < epsil)  count++;
				if (count > 3) break;
            }

            /*storing old values*/
			swap(P1, P1_prev);
			swap(P2, P2_prev);
			swap(P3, P3_prev);
            tk = tkp1;
        }

        if (epsil != 0.0f) free(Output_prev);
        free(P1); free(P2); free(P3); free(P1_prev); free(P2_prev); free(P3_prev); free(R1); free(R2); free(R3);
    }

    /*adding info into info_vector */
    infovector[0] = (float)(ll);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */

    return 0;
}
int apply_nonnegativity(float *A, long DimTotal)
{
	long i;
#pragma omp parallel for private(i)
	for (i = 0; i < DimTotal; i++)
	{
		A[i] = A[i] > 0.0f ? A[i] : 0;
	}

	return 1;
}
int calculate_norm(const float * A, const float * A_prev, float * re, long DimTotal)
{

	float re_temp = 0.0f;
	float re1_temp = 0.0f;

#pragma omp parallel
	{
		long i;
		float re_local = 0.0f;
		float re1_local = 0.0f;
		float diff;

#pragma omp for
		for (i = 0; i < DimTotal; i++)
		{
			diff = A[i] - A_prev[i];
			re_local += diff * diff;
			re1_local += A[i] * A[i];
		}

#pragma omp atomic
		re_temp += re_local;

#pragma omp atomic
		re1_temp += re1_local;
	}

	*re = sqrtf(re_temp) / sqrtf(re1_temp);

	return 1;
}

static inline float FD(long index, const float *D, long stride)
{
	return D[index] - D[index + stride];
}
int Grad_func(float *P1, float *P2, const float *D, const float *R1, const float *R2, float lambda,  long dimX, long dimY)
{
	float multip = (1.0f / (8.0f*lambda));

	long i, j, index;
	
#pragma omp parallel for private(i, j, index)
	for (j = 0; j < dimY - 1; j++)
	{
		index = j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			P1[index] = R1[index] + multip * FD(index, D, 1);
			P2[index] = R2[index] + multip * FD(index, D, dimX);
			index++;
		}

		//i = dimX - 1
		P1[index] = R1[index];
		P2[index] = R2[index] + multip * FD(index, D, dimX);
	}

	index = dimX * dimY - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		//j = dimY-1
		P1[index] = R1[index] + multip * FD(index, D, 1);
		P2[index] = R2[index];
		index++;
	}

	//i = dimX - 1, j = dimY-1, k = dimZ - 1
	P1[index] = R1[index];
	P2[index] = R2[index];

	return 1;
}
int Grad_func(float *P1, float *P2, float *P3, const float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ)
{
	float multip = (1.0f / (12.0f*lambda));

	long i, j, k, index;

	for (k = 0; k < dimZ - 1; k++)
	{

#pragma omp parallel for private(i, j, index)
		for (j = 0; j < dimY - 1; j++)
		{
			index = k * dimX * dimY + j * dimX;
			for (i = 0; i < dimX - 1; i++)
			{
				P1[index] = R1[index] + multip * FD(index, D, 1);
				P2[index] = R2[index] + multip * FD(index, D, dimX);
				P3[index] = R2[index] + multip * FD(index, D, dimX*dimY);
				index++;
			}

			//i = dimX - 1
			P1[index] = R1[index];
			P2[index] = R2[index] + multip * FD(index, D, dimX);
			P3[index] = R3[index] + multip * FD(index, D, dimX*dimY);
		}

		index = k * dimX * dimY + (dimY - 1) * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			P1[index] = R1[index] + multip * FD(index, D, 1);
			P2[index] = R2[index];
			P3[index] = R3[index] + multip * FD(index, D, dimX*dimY);
			index++;
		}

		//i = dimX - 1, j = dimY-1
		P1[index] = R1[index];
		P2[index] = R2[index];
		P3[index] = R3[index] + multip * FD(index, D, dimX*dimY);

	}

#pragma omp parallel for private(i, j, index)
	for (j = 0; j < dimY - 1; j++)
	{
		index = (dimZ - 1) * dimX * dimY + j * dimX;

		//k = dimZ - 1
		for (i = 0; i < dimX - 1; i++)
		{
			P1[index] = R1[index] + multip * FD(index, D, 1);
			P2[index] = R2[index] + multip * FD(index, D, dimX);
			P3[index] = R3[index];
			index++;
		}

		//i = dimX - 1, k = dimZ - 1
		P1[index] = R1[index];
		P2[index] = R2[index] + multip * FD(index, D, dimX);
		P3[index] = R3[index];
	}

	index = dimZ * dimY * dimX - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		//j = dimY-1, k = dimZ - 1
		P1[index] = R1[index] + multip * FD(index, D, 1);
		P2[index] = R2[index];
		P3[index] = R3[index];
		index++;
	}

	//i = dimX - 1, j = dimY-1, k = dimZ - 1
	index = dimZ * dimY * dimX - 1;
	P1[index] = R1[index];
	P2[index] = R2[index];
	P3[index] = R3[index];

	return 1;
}
int Rupd_func(const float *P1, const float *P1_old, const float *P2, const float *P2_old, float *R1, float *R2, float tkp1, float tk, long DimTotal)
{

	float multip = ((tk - 1.0f) / tkp1);
	float mulip_inv = 1 - multip;
	long i;

#pragma omp parallel for private(i)
	for (i = 0; i < DimTotal; i++)
	{
		R1[i] = P1[i] + multip * P1[i] + mulip_inv * P1_old[i];
		R2[i] = P2[i] + multip * P2[i] + mulip_inv * P2_old[i];
	}


	return 1;
}
int Rupd_func(const float *P1, const float *P1_old, const float *P2, const float *P2_old, const float *P3, const float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, long DimTotal)
{
	float multip = ((tk - 1.0f) / tkp1);
	float mulip_inv = 1 - multip;
	long i;

#pragma omp parallel for private(i)
	for (i = 0; i < DimTotal; i++)
	{
		R1[i] = P1[i] + multip * P1[i] + mulip_inv * P1_old[i];
		R2[i] = P2[i] + multip * P2[i] + mulip_inv * P2_old[i];
		R3[i] = P3[i] + multip * P3[i] + mulip_inv * P3_old[i];
	}

	return 1;
}
//inline functions for 2D boundary conditions
static inline float value(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] - R1[index - 1] + R2[index] - R2[index - dimX]);
}
static inline float value_i0(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] + R2[index] - R2[index - dimX]);
}
static inline float value_i1(long index, const float *R1, const float *R2, long dimX)
{
	return (-R1[index - 1] + R2[index] - R2[index - dimX]);
}
static inline float value_j0(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] - R1[index - 1] + R2[index]);
}
static inline float value_j1(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] - R1[index - 1] - R2[index - dimX]);
}
static inline float value_i0j0(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] + R2[index]);
}
static inline float value_i0j1(long index, const float *R1, const float *R2, long dimX)
{
	return (R1[index] - R2[index - dimX]);
}
static inline float value_i1j0(long index, const float *R1, const float *R2, long dimX)
{
	return (-R1[index - 1] + R2[index]);
}
static inline float value_i1j1(long index, const float *R1, const float *R2, long dimX)
{
	return (-R1[index - 1] - R2[index - dimX]);
}

int Obj_func(const float *A, float *D, const float *R1, const float *R2, float lambda, long dimX, long dimY)
{
	long i, j, index;

	//j = 0
	{
		index = 0;
		D[index] = A[index] - lambda * value_i0j0(index, R1, R2, dimX);
		index++;

		for (i = 1; i < dimX - 1; i++)
		{
			D[index] = A[index] - lambda * value_j0(index, R1, R2, dimX);
			index++;
		}

		D[index] = A[index] - lambda * value_i1j0(index, R1, R2, dimX);
	}

#pragma omp parallel for private(i, j, index)
	for (j = 1; j < dimY - 1; j++)
	{
		index = j * dimX;
		D[index] = A[index] - lambda * value_i0(index, R1, R2, dimX);
		index++;

		for (i = 1; i < dimX - 1; i++)
		{
			D[index] = A[index] - lambda * value(index, R1, R2, dimX);
			index++;
		}

		D[index] = A[index] - lambda * value_i1(index, R1, R2, dimX);
	}

	//j = dimY -1
	{
		index = (dimY - 1) * dimX;
		D[index] = A[index] - lambda * value_i0j1(index, R1, R2, dimX);
		index++;

		for (i = 1; i < dimX - 1; i++)
		{
			D[index] = A[index] - lambda * value_j1(index, R1, R2, dimX);
			index++;
		}

		D[index] = A[index] - lambda * value_i1j1(index, R1, R2, dimX);
	}


	return 1;
}

//inline functions for 3D bounday conditions
static inline float value(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_j0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_j1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] - R2[index - dimX] + R3[index]);
}
static inline float value_k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] - R2[index - dimX] - R3[index - dimX * dimY]);
}
static inline float value_i0j0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i0j1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i0k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] - R2[index - dimX] + R3[index]);
}
static inline float value_i0k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] - R2[index - dimX] - R3[index - dimX * dimY]);
}
static inline float value_i1j0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i1j1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] - R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}
static inline float value_i1k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] - R2[index - dimX] + R3[index]);
}
static inline float value_i1k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] - R2[index - dimX] + R3[index - dimX * dimY]);
}
static inline float value_j0k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] + R3[index]);
}
static inline float value_j0k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] + R2[index] - R3[index - dimX * dimY]);
}
static inline float value_j1k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] - R2[index - dimX] + R3[index]);
}
static inline float value_j1k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R1[index - 1] - R2[index - dimX] + R3[index - dimX * dimY]);
}
static inline float value_i0j0k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] - R3[index]);
}
static inline float value_i0j0k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] + R2[index] - R3[index - dimX * dimY]);
}
static inline float value_i0j1k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R2[index - dimX] + R3[index]);
}
static inline float value_i0j1k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (R1[index] - R2[index - dimX] - R3[index - dimX * dimY]);
}
static inline float value_i1j0k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] + R3[index]);
}
static inline float value_i1j0k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] + R2[index] - R3[index - dimX * dimY]);
}
static inline float value_i1j1k0(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R1[index - 1] - R2[index - dimX] + R3[index]);
}
static inline float value_i1j1k1(long index, const float *R1, const float *R2, const float * R3, long dimX, long dimY)
{
	return (-R2[index - dimX] + R3[index] - R3[index - dimX * dimY]);
}


int Obj_func(const float *A, float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ)
{

	long i, j, k, index;

	//k = 0
	{
		//j = 0
		{
			index = 0;
			D[index] = A[index] - lambda * value_i0j0k0(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j0k0(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j0k0(index, R1, R2, R3, dimX, dimY);
		}

#pragma omp parallel for private(i, j, index)
		for (j = 1; j < dimY - 1; j++)
		{
			index = j * dimX;
			D[index] = A[index] - lambda * value_i0k0(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_k0(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1k0(index, R1, R2, R3, dimX, dimY);
		}

		//j = dimY -1
		{
			index = (dimY - 1) * dimX;
			D[index] = A[index] - lambda * value_i0j1k0(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j1k0(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j1k0(index, R1, R2, R3, dimX, dimY);
		}

	}

	for (k = 1; k < dimZ - 1; k++)
	{
		//j = 0
		{
			index = k * dimX *dimY;
			D[index] = A[index] - lambda * value_i0j0(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j0(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j0(index, R1, R2, R3, dimX, dimY);
		}

#pragma omp parallel for private(i, j, index)
		for (j = 1; j < dimY - 1; j++)
		{
			index = k * dimX *dimY + j * dimX;
			D[index] = A[index] - lambda * value_i0(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1(index, R1, R2, R3, dimX, dimY);
		}

		//j = dimY -1
		{
			index = k * dimX *dimY + (dimY - 1) * dimX;
			D[index] = A[index] - lambda * value_i0j1(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j1(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j1(index, R1, R2, R3, dimX, dimY);
		}

	}

	//k = dimZ -1
	{
		//j = 0
		{
			index = (dimZ - 1) * dimX *dimY;
			D[index] = A[index] - lambda * value_i0j0k1(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j0k1(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j0k1(index, R1, R2, R3, dimX, dimY);
		}

#pragma omp parallel for private(i, j, index)
		for (j = 1; j < dimY - 1; j++)
		{
			index = (dimZ - 1) * dimX *dimY + j * dimX;
			D[index] = A[index] - lambda * value_i0k1(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_k1(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1k1(index, R1, R2, R3, dimX, dimY);
		}

		//j = dimY -1
		{
			//i = 0
			index = (dimZ - 1) * dimX *dimY + (dimY - 1) * dimX;
			D[index] = A[index] - lambda * value_i0j1k1(index, R1, R2, R3, dimX, dimY);
			index++;

			for (i = 1; i < dimX - 1; i++)
			{
				D[index] = A[index] - lambda * value_j1k1(index, R1, R2, R3, dimX, dimY);
				index++;
			}

			D[index] = A[index] - lambda * value_i1j1k1(index, R1, R2, R3, dimX, dimY);
		}

	}

	return 1;
}

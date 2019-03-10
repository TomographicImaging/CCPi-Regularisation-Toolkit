 /*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev
Copyright 2019 Srikanth Nagella, Edoardo Pasca

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

#include "TGV_GPU_core.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>


/* CUDA implementation of Primal-Dual denoising method for
 * Total Generilized Variation (TGV)-L2 model [1] (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume (2D/3D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 * 7. eplsilon: tolerance constant
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]

 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */


#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void DualP_2D_kernel(float *U, float *V1, float *V2, float *P1, float *P2, long dimX, long dimY, float sigma)
{
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {
        /* symmetric boundary conditions (Neuman) */
        if ((i >= 0) && (i < dimX-1))  P1[index] += sigma*((U[(i+1) + dimX*j] - U[index])  - V1[index]);
        else if  (i == dimX-1) P1[index] -= sigma*(V1[index]);
        else P1[index] = 0.0f;
        if ((j >= 0) && (j < dimY-1))  P2[index] += sigma*((U[i + dimX*(j+1)] - U[index])  - V2[index]);
        else if  (j == dimY-1) P2[index] -= sigma*(V2[index]);
        else P2[index] = 0.0f;
	}
	return;
}

__global__ void ProjP_2D_kernel(float *P1, float *P2, long dimX, long dimY, float alpha1)
{
   	float grad_magn;

	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {
            grad_magn = sqrtf(powf(P1[index],2) + powf(P2[index],2));
            grad_magn = grad_magn/alpha1;
            if (grad_magn > 1.0f) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
            }
	}
	return;
}

__global__ void DualQ_2D_kernel(float *V1, float *V2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float sigma)
{
        float q1, q2, q11, q22;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {
         q1 = 0.0f; q2  = 0.0f; q11  = 0.0f; q22  = 0.0f;

	        if ((i >= 0) && (i < dimX-1))  {
        	    /* boundary conditions (Neuman) */
        	    q1 = V1[(i+1) + dimX*j] - V1[index];
        	    q11 = V2[(i+1) + dimX*j] - V2[index];
	        }
        	if ((j >= 0) && (j < dimY-1)) {
        	    q2 = V2[i + dimX*(j+1)] - V2[index];
        	    q22 = V1[i + dimX*(j+1)] - V1[index];
        	}

            Q1[index] += sigma*(q1);
            Q2[index] += sigma*(q2);
            Q3[index] += sigma*(0.5f*(q11 + q22));
	 }
	return;
}

__global__ void ProjQ_2D_kernel(float *Q1, float *Q2, float *Q3, long dimX, long dimY, float alpha0)
{
	float grad_magn;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {
            grad_magn = sqrt(powf(Q1[index],2) + powf(Q2[index],2) + 2*powf(Q3[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0f) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
        	    }
	}
	return;
}

__global__ void DivProjP_2D_kernel(float *U, float *U0, float *P1, float *P2, long dimX, long dimY, float lambda, float tau)
{
	float P_v1, P_v2, div;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {

        if ((i > 0) && (i < dimX-1)) P_v1 = P1[index] - P1[(i-1) + dimX*j];
        else if (i == dimX-1) P_v1 = -P1[(i-1) + dimX*j];
        else if (i == 0) P_v1 = P1[index];
        else P_v1 = 0.0f;

      	if ((j > 0) && (j < dimY-1))  P_v2 = P2[index] - P2[i + dimX*(j-1)];
      	else if (j == dimY-1) P_v2 = -P2[i + dimX*(j-1)];
      	else if (j == 0) P_v2 = P2[index];
      	else P_v2 = 0.0f;


        div = P_v1 + P_v2;
        U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
	}
	return;
}

__global__ void UpdV_2D_kernel(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float tau)
{
	float q1, q3_x, q2, q3_y, div1, div2;
	long i1, j1;

	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

        if ((i < dimX) && (j < dimY)) {

	q1 = 0.0f; q3_x = 0.0f; q2 = 0.0f; q3_y = 0.0f; div1 = 0.0f; div2= 0.0f;

	    i1 = (i-1) + dimX*j;
            j1 = (i) + dimX*(j-1);

            /* boundary conditions (Neuman) */
            if ((i > 0) && (i < dimX-1)) {
            q1 = Q1[index] - Q1[i1];
            q3_x = Q3[index] - Q3[i1];  }
            else if (i == 0) {
            q1 = Q1[index];
            q3_x = Q3[index]; }
            else if (i == dimX-1) {
            q1 = -Q1[i1];
            q3_x = -Q3[i1];  }
            else {
            q1 = 0.0f;
            q3_x = 0.0f;
            }

            if ((j > 0) && (j < dimY-1)) {
            q2 = Q2[index] - Q2[j1];
            q3_y = Q3[index] - Q3[j1]; }
            else if (j == dimY-1) {
            q2 = -Q2[j1];
            q3_y = -Q3[j1]; }
            else if (j == 0) {
            q2 = Q2[index];
            q3_y = Q3[index]; }
            else {
            q2 = 0.0f;
            q3_y = 0.0f;
            }

            div1 = q1 + q3_y;
            div2 = q3_x + q2;
            V1[index] += tau*(P1[index] + div1);
            V2[index] += tau*(P2[index] + div2);
	}
	return;
}

__global__ void copyIm_TGV_kernel(float *U, float *U_old, long dimX, long dimY)
{
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

    if ((i < dimX) && (j < dimY)) {
        U_old[index] = U[index];
    }
}

__global__ void copyIm_TGV_kernel_ar2(float *V1, float *V2, float *V1_old, float *V2_old, long dimX, long dimY)
{
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

    if ((i < dimX) && (j < dimY)) {
        V1_old[index] = V1[index];
        V2_old[index] = V2[index];
    }
}

__global__ void newU_kernel(float *U, float *U_old, long dimX, long dimY)
{
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

    if ((i < dimX) && (j < dimY)) {
        U[index] = 2.0f*U[index] - U_old[index];
    }
}


__global__ void newU_kernel_ar2(float *V1, float *V2, float *V1_old, float *V2_old, long dimX, long dimY)
{
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

    if ((i < dimX) && (j < dimY)) {
        V1[index] = 2.0f*V1[index] - V1_old[index];
        V2[index] = 2.0f*V2[index] - V2_old[index];
    }
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
__global__ void DualP_3D_kernel(float *U, float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float sigma)
{
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;

        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
            /* symmetric boundary conditions (Neuman) */
            if ((i >= 0) && (i < dimX-1)) P1[index] += sigma*((U[(dimX*dimY)*k + (i+1)*dimX+j] - U[index])  - V1[index]);
	    else if (i == dimX-1) P1[index] -= sigma*(V1[index]);
	    else P1[index] = 0.0f;
	    if ((j >= 0) && (j < dimY-1)) P2[index] += sigma*((U[(dimX*dimY)*k + i*dimX+(j+1)] - U[index])  - V2[index]);
	    else if (j == dimY-1) P2[index] -= sigma*(V2[index]);
	    else P2[index] = 0.0f;
      	    if ((k >= 0) && (k < dimZ-1)) P3[index] += sigma*((U[(dimX*dimY)*(k+1) + i*dimX+(j)] - U[index])  - V3[index]);
      	    else if (k == dimZ-1) P3[index] -= sigma*(V3[index]);
      	    else P3[index] = 0.0f;
	 }
	return;
}

__global__ void ProjP_3D_kernel(float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float alpha1)
{
   	float grad_magn;
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;
        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
            grad_magn = (sqrtf(powf(P1[index],2) + powf(P2[index],2) + powf(P3[index],2)))/alpha1;
            if (grad_magn > 1.0f) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
                P3[index] /= grad_magn;
            }
	}
	return;
}

__global__ void DualQ_3D_kernel(float *V1, float *V2, float *V3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float sigma)
{

        float q1, q2, q3, q11, q22, q33, q44, q55, q66;
	long index;

	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;
        long i1 = (dimX*dimY)*k + (i+1)*dimX+j;
        long j1 = (dimX*dimY)*k + (i)*dimX+(j+1);
        long k1 = (dimX*dimY)*(k+1) + (i)*dimX+(j);

        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
 	q1 = 0.0f; q11 = 0.0f; q33 = 0.0f; q2 = 0.0f; q22 = 0.0f; q55 = 0.0f; q3 = 0.0f; q44 = 0.0f; q66 = 0.0f;

	        /* boundary conditions (Neuman) */
	        if ((i >= 0) && (i < dimX-1))  {
                q1 = V1[i1] - V1[index];
                q11 = V2[i1] - V2[index];
                q33 = V3[i1] - V3[index];  }
        	if ((j >= 0) && (j < dimY-1)) {
                q2 = V2[j1] - V2[index];
                q22 = V1[j1] - V1[index];
                q55 = V3[j1] - V3[index];  }
        	if ((k >= 0) && (k < dimZ-1)) {
                q3 = V3[k1] - V3[index];
                q44 = V1[k1] - V1[index];
                q66 = V2[k1] - V2[index]; }

            Q1[index] += sigma*(q1); /*Q11*/
            Q2[index] += sigma*(q2); /*Q22*/
            Q3[index] += sigma*(q3); /*Q33*/
            Q4[index] += sigma*(0.5f*(q11 + q22)); /* Q21 / Q12 */
            Q5[index] += sigma*(0.5f*(q33 + q44)); /* Q31 / Q13 */
            Q6[index] += sigma*(0.5f*(q55 + q66)); /* Q32 / Q23 */
	 }
	return;
}

__global__ void ProjQ_3D_kernel(float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float alpha0)
{
	float grad_magn;
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;

        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
	grad_magn = sqrtf(powf(Q1[index],2) + powf(Q2[index],2) + powf(Q3[index],2) + 2.0f*powf(Q4[index],2) + 2.0f*powf(Q5[index],2) + 2.0f*powf(Q6[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0f) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
                Q4[index] /= grad_magn;
                Q5[index] /= grad_magn;
                Q6[index] /= grad_magn;
            }
	}
	return;
}
__global__ void DivProjP_3D_kernel(float *U, float *U0, float *P1, float *P2, float *P3, long dimX, long dimY, long dimZ, float lambda, float tau)
{
	float P_v1, P_v2, P_v3, div;
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;
        long i1 = (dimX*dimY)*k + (i-1)*dimX+j;
        long j1 = (dimX*dimY)*k + (i)*dimX+(j-1);
        long k1 = (dimX*dimY)*(k-1) + (i)*dimX+(j);

        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {

        if ((i > 0) && (i < dimX-1)) P_v1 = P1[index] - P1[i1];
        else if (i == dimX-1) P_v1 = -P1[i1];
        else if (i == 0) P_v1 = P1[index];
        else P_v1 = 0.0f;

      	if ((j > 0) && (j < dimY-1))  P_v2 = P2[index] - P2[j1];
      	else if (j == dimY-1) P_v2 = -P2[j1];
        else if (j == 0) P_v2 = P2[index];
        else P_v2 = 0.0f;

      	if ((k > 0) && (k < dimZ-1))  P_v3 = P3[index] - P3[k1];
      	else if (k == dimZ-1) P_v3 = -P3[k1];
        else if (k == 0) P_v3 = P3[index];
        else P_v3 = 0.0f;

        div = P_v1 + P_v2 + P_v3;
        U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
	}
	return;
}
__global__ void UpdV_3D_kernel(float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, long dimX, long dimY, long dimZ, float tau)
{
	float q1, q4x, q5x, q2, q4y, q6y, q6z, q5z, q3, div1, div2, div3;
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        index = (dimX*dimY)*k + i*dimX+j;
        long i1 = (dimX*dimY)*k + (i-1)*dimX+j;
        long j1 = (dimX*dimY)*k + (i)*dimX+(j-1);
        long k1 = (dimX*dimY)*(k-1) + (i)*dimX+(j);

        /* Q1 - Q11, Q2 - Q22, Q3 -  Q33, Q4 - Q21/Q12, Q5 - Q31/Q13, Q6 - Q32/Q23*/
        if ((i < dimX) && (j < dimY)  && (k < dimZ)) {

  	 /* boundary conditions (Neuman) */
            if ((i > 0) && (i < dimX-1)) {
                q1 = Q1[index] - Q1[i1];
                q4x = Q4[index] - Q4[i1];
                q5x = Q5[index] - Q5[i1]; }
            else if (i == 0) {
                q1 = Q1[index];
                q4x = Q4[index];
                q5x = Q5[index]; }
            else if (i == dimX-1) {
                q1 = -Q1[i1];
                q4x = -Q4[i1];
                q5x = -Q5[i1]; }
            else {
                q1 = 0.0f;
                q4x = 0.0f;
                q5x = 0.0f;  }

            if ((j > 0) && (j < dimY-1)) {
                q2 = Q2[index] - Q2[j1];
                q4y = Q4[index] - Q4[j1];
                q6y = Q6[index] - Q6[j1]; }
            else if (j == dimY-1) {
                q2 = -Q2[j1];
                q4y = -Q4[j1];
                q6y = -Q6[j1]; }
            else if (j == 0) {
                q2 = Q2[index];
                q4y = Q4[index];
                q6y = Q6[index]; }
            else {
                q2 =  0.0f;
                q4y = 0.0f;
                q6y = 0.0f;
               }

            if ((k > 0) && (k < dimZ-1)) {
                q6z = Q6[index] - Q6[k1];
                q5z = Q5[index] - Q5[k1];
                q3 = Q3[index] - Q3[k1]; }
            else if (k == dimZ-1) {
                q6z = -Q6[k1];
                q5z = -Q5[k1];
                q3 = -Q3[k1]; }
            else if (k == 0) {
                q6z = Q6[index];
                q5z = Q5[index];
                q3 = Q3[index]; }
            else {
                q6z = 0.0f;
                q5z = 0.0f;
                q3 = 0.0f; }

       div1 = q1 + q4y + q5z;
       div2 = q4x + q2 + q6z;
       div3 = q5x + q6y + q3;

        V1[index] += tau*(P1[index] + div1);
        V2[index] += tau*(P2[index] + div2);
        V3[index] += tau*(P3[index] + div3);
	}
	return;
}

__global__ void copyIm_TGV_kernel3D(float *U, float *U_old, long dimX, long dimY, long dimZ)
{
    long index;
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    index = (dimX*dimY)*k + j*dimX+i;

    if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
      	U_old[index] = U[index];
    }
}

__global__ void copyIm_TGV_kernel3D_ar3(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, long dimX, long dimY, long dimZ)
{
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

    index = (dimX*dimY)*k + j*dimX+i;

    if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
      	V1_old[index] = V1[index];
	V2_old[index] = V2[index];
	V3_old[index] = V3[index];
    }
}

__global__ void newU_kernel3D(float *U, float *U_old, long dimX, long dimY, long dimZ)
{
	long index;
	const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

     index = (dimX*dimY)*k + j*dimX+i;

    if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
	   U[index] = 2.0f*U[index] - U_old[index];
    }
}

__global__ void newU_kernel3D_ar3(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, long dimX, long dimY, long dimZ)
{
	 long index;
	 const long i = blockDim.x * blockIdx.x + threadIdx.x;
   const long j = blockDim.y * blockIdx.y + threadIdx.y;
   const long k = blockDim.z * blockIdx.z + threadIdx.z;

     index = (dimX*dimY)*k + j*dimX+i;

    if ((i < dimX) && (j < dimY)  && (k < dimZ)) {
	   V1[index] = 2.0f*V1[index] - V1_old[index];
	   V2[index] = 2.0f*V2[index] - V2_old[index];
	   V3[index] = 2.0f*V3[index] - V3_old[index];
    }
}

__global__ void TGVResidCalc2D_kernel(float *Input1, float *Input2, float* Output, long dimX, long dimY, long num_total)
{
      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;

        long index = i + (dimX)*j;

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

__global__ void TGVResidCalc3D_kernel(float *Input1, float *Input2, float* Output, long dimX, long dimY, long dimZ, long num_total)
{
  long index;
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

    index = (dimX*dimY)*k + j*dimX+i;

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/************************ MAIN HOST FUNCTION ***********************/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
extern "C" int TGV_GPU_main(float *U0, float *U, float *infovector, float lambda, float alpha1, float alpha0, int iterationsNumb, float L2, float epsil, int dimX, int dimY, int dimZ)
{

        int deviceCount = -1; // number of devices
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
         fprintf(stderr, "No CUDA devices found\n");
        return -1;
        }

	      long dimTotal = (long)(dimX*dimY*dimZ);
        float *U_old, *d_U0, *d_U, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, tau, sigma, re;
        int n, count;
        count = 0; re = 0.0f;
        tau = powf(L2,-0.5f);
        sigma = tau;

        CHECK(cudaMalloc((void**)&d_U0,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_U,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&U_old,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&P1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&P2,dimTotal*sizeof(float)));

        CHECK(cudaMalloc((void**)&Q1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q2,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V2,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V1_old,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V2_old,dimTotal*sizeof(float)));

        CHECK(cudaMemcpy(d_U0,U0,dimTotal*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_U,U0,dimTotal*sizeof(float),cudaMemcpyHostToDevice));
        cudaMemset(P1, 0, dimTotal*sizeof(float));
        cudaMemset(P2, 0, dimTotal*sizeof(float));
        cudaMemset(Q1, 0, dimTotal*sizeof(float));
        cudaMemset(Q2, 0, dimTotal*sizeof(float));
        cudaMemset(Q3, 0, dimTotal*sizeof(float));
        cudaMemset(V1, 0, dimTotal*sizeof(float));
        cudaMemset(V2, 0, dimTotal*sizeof(float));

        if (dimZ == 1) {
	/*2D case */
	dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
	dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));

        for(n=0; n < iterationsNumb; n++) {

	    /* Calculate Dual Variable P */
            DualP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, V1, V2, P1, P2, (long)(dimX), (long)(dimY), sigma);
      	    checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*Projection onto convex set for P*/
            ProjP_2D_kernel<<<dimGrid,dimBlock>>>(P1, P2, (long)(dimX), (long)(dimY), alpha1);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /* Calculate Dual Variable Q */
            DualQ_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, Q1, Q2, Q3, (long)(dimX), (long)(dimY), sigma);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*Projection onto convex set for Q*/
            ProjQ_2D_kernel<<<dimGrid,dimBlock>>>(Q1, Q2, Q3, (long)(dimX), (long)(dimY), alpha0);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*saving U into U_old*/
            copyIm_TGV_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, (long)(dimX), (long)(dimY));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, d_U0, P1, P2, (long)(dimX), (long)(dimY), lambda, tau);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*get updated solution U*/
            newU_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, (long)(dimX), (long)(dimY));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*saving V into V_old*/
            copyIm_TGV_kernel_ar2<<<dimGrid,dimBlock>>>(V1, V2, V1_old, V2_old, (long)(dimX), (long)(dimY));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /* upd V*/
            UpdV_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, P1, P2, Q1, Q2, Q3, (long)(dimX), (long)(dimY), tau);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*get new V*/
            newU_kernel_ar2<<<dimGrid,dimBlock>>>(V1, V2, V1_old, V2_old, (long)(dimX), (long)(dimY));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );

            if ((epsil != 0.0f) && (n % 5 == 0)) {
                /* calculate norm - stopping rules using the Thrust library */
                TGVResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, V1_old, (long)(dimX), (long)(dimY), dimTotal);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );

                // setup arguments
                square<float>        unary_op;
                thrust::plus<float> binary_op;
                thrust::device_vector<float> d_vec(V1_old, V1_old + dimTotal);
                float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
                thrust::device_vector<float> d_vec2(d_U, d_U + dimTotal);
                float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

                // compute norm
                re = (reduction/reduction2);
                if (re < epsil)  count++;
                if (count > 3) break;
          }
	    }
        }
        else {
        /*3D case */
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKXSIZE));

        float *P3, *Q4, *Q5, *Q6, *V3, *V3_old;

	      CHECK(cudaMalloc((void**)&P3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q4,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q5,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q6,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V3_old,dimTotal*sizeof(float)));

        cudaMemset(Q4, 0.0f, dimTotal*sizeof(float));
        cudaMemset(Q5, 0.0f, dimTotal*sizeof(float));
        cudaMemset(Q6, 0.0f, dimTotal*sizeof(float));
        cudaMemset(P3, 0.0f, dimTotal*sizeof(float));
        cudaMemset(V3, 0.0f, dimTotal*sizeof(float));

        for(n=0; n < iterationsNumb; n++) {

	    /* Calculate Dual Variable P */
            DualP_3D_kernel<<<dimGrid,dimBlock>>>(d_U, V1, V2, V3, P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), sigma);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*Projection onto convex set for P*/
            ProjP_3D_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), alpha1);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /* Calculate Dual Variable Q */
            DualQ_3D_kernel<<<dimGrid,dimBlock>>>(V1, V2, V3, Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), sigma);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
             /*Projection onto convex set for Q*/
            ProjQ_3D_kernel<<<dimGrid,dimBlock>>>(Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), alpha0);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*saving U into U_old*/
            copyIm_TGV_kernel3D<<<dimGrid,dimBlock>>>(d_U, U_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_3D_kernel<<<dimGrid,dimBlock>>>(d_U, d_U0, P1, P2, P3, (long)(dimX), (long)(dimY), (long)(dimZ), lambda, tau);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*get updated solution U*/
            newU_kernel3D<<<dimGrid,dimBlock>>>(d_U, U_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*saving V into V_old*/
            copyIm_TGV_kernel3D_ar3<<<dimGrid,dimBlock>>>(V1, V2, V3, V1_old, V2_old, V3_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /* upd V*/
            UpdV_3D_kernel<<<dimGrid,dimBlock>>>(V1, V2, V3, P1, P2, P3, Q1, Q2, Q3, Q4, Q5, Q6, (long)(dimX), (long)(dimY), (long)(dimZ), tau);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            /*get new V*/
            newU_kernel3D_ar3<<<dimGrid,dimBlock>>>(V1, V2, V3, V1_old, V2_old, V3_old, (long)(dimX), (long)(dimY), (long)(dimZ));
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );

            if ((epsil != 0.0f) && (n % 5 == 0)) {
                /* calculate norm - stopping rules using the Thrust library */
                TGVResidCalc3D_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, V1_old, (long)(dimX), (long)(dimY), (long)(dimZ), dimTotal);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );

                // setup arguments
                square<float>        unary_op;
                thrust::plus<float> binary_op;
                thrust::device_vector<float> d_vec(V1_old, V1_old + dimTotal);
                float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
                thrust::device_vector<float> d_vec2(d_U, d_U + dimTotal);
                float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

                // compute norm
                re = (reduction/reduction2);
                if (re < epsil)  count++;
                if (count > 3) break;
              }
	       }

        CHECK(cudaFree(Q4));
        CHECK(cudaFree(Q5));
        CHECK(cudaFree(Q6));
        CHECK(cudaFree(P3));
        CHECK(cudaFree(V3));
        CHECK(cudaFree(V3_old));
        }

        CHECK(cudaMemcpy(U,d_U,dimTotal*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_U0));
        CHECK(cudaFree(d_U));
        CHECK(cudaFree(U_old));
        CHECK(cudaFree(P1));
        CHECK(cudaFree(P2));

        CHECK(cudaFree(Q1));
        CHECK(cudaFree(Q2));
        CHECK(cudaFree(Q3));
        CHECK(cudaFree(V1));
        CHECK(cudaFree(V2));
        CHECK(cudaFree(V1_old));
        CHECK(cudaFree(V2_old));

        //cudaDeviceReset();
        /*adding info into info_vector */
        infovector[0] = (float)(n);  /*iterations number (if stopped earlier based on tolerance)*/
        infovector[1] = re;  /* reached tolerance */

        return 0;
}

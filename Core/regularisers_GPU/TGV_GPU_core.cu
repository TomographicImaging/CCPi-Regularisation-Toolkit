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

#include "TGV_GPU_core.h"

/* CUDA implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model [1] (2D case only)
 *
 * Input Parameters:
 * 1. Noisy image (2D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 *
 * Output:
 * Filtered/regulariaed image 
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return;                                                               \
    }                                                                          \
}
    
    
#define BLKXSIZE2D 16
#define BLKYSIZE2D 16
#define EPS 1.0e-7
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void DualP_2D_kernel(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, float sigma)
{
    
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[index] += sigma*((U[j*dimX+(i-1)] - U[index]) - V1[index]); 
            else P1[index] += sigma*((U[j*dimX+(i+1)] - U[index])  - V1[index]); 
            if (j == dimY-1) P2[index] += sigma*((U[(j-1)*dimX+i] - U[index])  - V2[index]);
            else  P2[index] += sigma*((U[(j+1)*dimX+i] - U[index])  - V2[index]);
		}
	return;
} 

__global__ void ProjP_2D_kernel(float *P1, float *P2, int dimX, int dimY, float alpha1)
{
    float grad_magn;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
            grad_magn = sqrt(pow(P1[index],2) + pow(P2[index],2));
            grad_magn = grad_magn/alpha1;
            if (grad_magn > 1.0) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
            }
		}
	return;
} 

__global__ void DualQ_2D_kernel(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, float sigma)
{
        float q1, q2, q11, q22;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1)
            { q1 = (V1[j*dimX+(i-1)] - V1[index]);
              q11 = (V2[j*dimX+(i-1)] - V2[index]);
            }
            else {
                q1 = (V1[j*dimX+(i+1)] - V1[index]);
                q11 = (V2[j*dimX+(i+1)] - V2[index]);
            }
            if (j == dimY-1) {
                q2 = (V2[(j-1)*dimX+i] - V2[index]);
                q22 = (V1[(j-1)*dimX+i] - V1[index]);
            }
            else {
                q2 = V2[(j+1)*dimX+i] - V2[index];
                q22 = V1[(j+1)*dimX+i] - V1[index];
            }
            Q1[index] += sigma*(q1);
            Q2[index] += sigma*(q2);
            Q3[index] += sigma*(0.5f*(q11 + q22));
		}
	return;
} 

__global__ void ProjQ_2D_kernel(float *Q1, float *Q2, float *Q3, int dimX, int dimY, float alpha0)
{
    float grad_magn;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
            grad_magn = sqrt(pow(Q1[index],2) + pow(Q2[index],2) + 2*pow(Q3[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
            }
		}
	return;
} 

__global__ void DivProjP_2D_kernel(float *U, float *U0, float *P1, float *P2, int dimX, int dimY, float lambda, float tau)
{
		float P_v1, P_v2, div;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
			
            if (i == 0) P_v1 = P1[index];
            else P_v1 = P1[index] - P1[j*dimX+(i-1)];
            if (j == 0) P_v2 = P2[index];
            else  P_v2 = P2[index] - P2[(j-1)*dimX+i];
            div = P_v1 + P_v2;
            U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
		}
	return;
} 

__global__ void UpdV_2D_kernel(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, float tau)
{
		float q1, q11, q2, q22, div1, div2;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
			
   /* symmetric boundary conditions (Neuman) */
            if (i == 0) {
                q1 = Q1[index];
                q11 = Q3[index];
            }
            else {
                q1 = Q1[index] - Q1[j*dimX+(i-1)];
                q11 = Q3[index] - Q3[j*dimX+(i-1)];
            }
            if (j == 0) {
                q2 = Q2[index];
                q22 = Q3[index];
            }
            else  {
                q2 = Q2[index] - Q2[(j-1)*dimX+i];
                q22 = Q3[index] - Q3[(j-1)*dimX+i];
            }
            div1 = q1 + q22;
            div2 = q2 + q11;
            V1[index] += tau*(P1[index] + div1);
            V2[index] += tau*(P2[index] + div2);
		}
	return;
} 

__global__ void copyIm_TGV_kernel(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

__global__ void newU_kernel(float *U, float *U_old, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        U[index] = 2.0f*U[index] - U_old[index]; 
    }
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/********************* MAIN HOST FUNCTION ******************/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
extern "C" void TGV_GPU_main(float *U0, float *U, float lambda, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY)
{
		int dimTotal, dev = 0;
		CHECK(cudaSetDevice(dev));
		dimTotal = dimX*dimY;
       
        float *U_old, *d_U0, *d_U, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, tau, sigma;
        tau = pow(L2,-0.5);
        sigma = pow(L2,-0.5);
                                      
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
        
	     /*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));
             
        for(int n=0; n < iterationsNumb; n++) {
			
		    /* Calculate Dual Variable P */
            DualP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, V1, V2, P1, P2, dimX, dimY, sigma);
			CHECK(cudaDeviceSynchronize());
			/*Projection onto convex set for P*/
            ProjP_2D_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, alpha1);
            CHECK(cudaDeviceSynchronize());
            /* Calculate Dual Variable Q */
            DualQ_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, Q1, Q2, Q3, dimX, dimY, sigma);
            CHECK(cudaDeviceSynchronize());
             /*Projection onto convex set for Q*/
            ProjQ_2D_kernel<<<dimGrid,dimBlock>>>(Q1, Q2, Q3, dimX, dimY, alpha0);
            CHECK(cudaDeviceSynchronize());
            /*saving U into U_old*/
            copyIm_TGV_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, d_U0, P1, P2, dimX, dimY, lambda, tau);
            CHECK(cudaDeviceSynchronize());
            /*get updated solution U*/
            newU_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*saving V into V_old*/
            copyIm_TGV_kernel<<<dimGrid,dimBlock>>>(V1, V1_old, dimX, dimY, dimTotal);
            copyIm_TGV_kernel<<<dimGrid,dimBlock>>>(V2, V2_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /* upd V*/
            UpdV_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, P1, P2, Q1, Q2, Q3, dimX, dimY, tau);
            CHECK(cudaDeviceSynchronize());
            /*get new V*/
            newU_kernel<<<dimGrid,dimBlock>>>(V1, V1_old, dimX, dimY, dimTotal);
            newU_kernel<<<dimGrid,dimBlock>>>(V2, V2_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());            
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
}

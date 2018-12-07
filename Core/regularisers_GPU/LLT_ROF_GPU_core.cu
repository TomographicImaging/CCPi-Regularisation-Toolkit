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

#include "LLT_ROF_GPU_core.h"

/* CUDA implementation of Lysaker, Lundervold and Tai (LLT) model [1] combined with Rudin-Osher-Fatemi [2] TV regularisation penalty.
 * 
* This penalty can deliver visually pleasant piecewise-smooth recovery if regularisation parameters are selected well. 
* The rule of thumb for selection is to start with lambdaLLT = 0 (just the ROF-TV model) and then proceed to increase 
* lambdaLLT starting with smaller values. 
*
* Input Parameters:
* 1. U0 - original noise image/volume
* 2. lambdaROF - ROF-related regularisation parameter
* 3. lambdaLLT - LLT-related regularisation parameter
* 4. tau - time-marching step 
* 5. iter - iterations number (for both models)
*
* Output:
* Filtered/regularised image
*
* References: 
* [1] Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.
* [2] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
*/

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                 /                                         \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return;                                                               \
    }                                                                          \
}
    
#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8
    
#define BLKXSIZE2D 16
#define BLKYSIZE2D 16


#define EPS_LLT 0.01
#define EPS_ROF 1.0e-12

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__host__ __device__ int signLLT (float x)
{
        return (x > 0) - (x < 0);
}        
   
/*************************************************************************/
/**********************LLT-related functions *****************************/
/*************************************************************************/
__global__ void der2D_LLT_kernel(float *U, float *D1, float *D2, int dimX, int dimY)
    {
		int i_p, i_m, j_m, j_p;
		float dxx, dyy, denom_xx, denom_yy;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
			/* symmetric boundary conditions (Neuman) */
			i_p = i + 1; if (i_p == dimX) i_p = i - 1;
			i_m = i - 1; if (i_m < 0) i_m = i + 1;
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;

			dxx = U[j*dimX+i_p] - 2.0f*U[index] + U[j*dimX+i_m];
			dyy = U[j_p*dimX+i] - 2.0f*U[index] + U[j_m*dimX+i];

			denom_xx = abs(dxx) + EPS_LLT;
			denom_yy = abs(dyy) + EPS_LLT;

			D1[index] = dxx / denom_xx;
			D2[index] = dyy / denom_yy;
		}
	}
	
__global__ void der3D_LLT_kernel(float* U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ)
    {
		int i_p, i_m, j_m, j_p, k_p, k_m;
		float dxx, dyy, dzz, denom_xx, denom_yy, denom_zz;
		
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {
			
        /* symmetric boundary conditions (Neuman) */
 		i_p = i + 1; if (i_p == dimX) i_p = i - 1;
 		i_m = i - 1; if (i_m < 0) i_m = i + 1;
 		j_p = j + 1; if (j_p == dimY) j_p = j - 1;
 		j_m = j - 1; if (j_m < 0) j_m = j + 1;
 		k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
 		k_m = k - 1; if (k_m < 0) k_m = k + 1;
        
      	int index = (dimX*dimY)*k + j*dimX+i;
      	
      	dxx = U[(dimX*dimY)*k + j*dimX+i_p] - 2.0f*U[index] + U[(dimX*dimY)*k + j*dimX+i_m];
 		dyy = U[(dimX*dimY)*k + j_p*dimX+i] - 2.0f*U[index] + U[(dimX*dimY)*k + j_m*dimX+i];
 		dzz = U[(dimX*dimY)*k_p + j*dimX+i] - 2.0f*U[index] + U[(dimX*dimY)*k_m + j*dimX+i];
 
 		denom_xx = abs(dxx) + EPS_LLT;
 		denom_yy = abs(dyy) + EPS_LLT;
 		denom_zz = abs(dzz) + EPS_LLT;
 
 		D1[index] = dxx / denom_xx;
 		D2[index] = dyy / denom_yy;
 		D3[index] = dzz / denom_zz;
		}
	}

/*************************************************************************/
/**********************ROF-related functions *****************************/
/*************************************************************************/

/* first-order differences 1 */
__global__ void D1_func2D_ROF_kernel(float* Input, float* D1, int N, int M)
    {
		int i1, j1, i2;
		float NOMx_1,NOMy_1,NOMy_0,denom1,denom2,T1;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + N*j;        
        
        if ((i >= 0) && (i < N) && (j >= 0) && (j < M)) {
            
            /* boundary conditions (Neumann reflections) */
                i1 = i + 1; if (i1 >= N) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= M) j1 = j-1;
		
		     /* Forward-backward differences */
                NOMx_1 = Input[j1*N + i] - Input[index]; /* x+ */
                NOMy_1 = Input[j*N + i1] - Input[index]; /* y+ */                
                NOMy_0 = Input[index] - Input[j*N + i2]; /* y- */
                
                denom1 = NOMx_1*NOMx_1;
                denom2 = 0.5f*(signLLT((float)NOMy_1) + signLLT((float)NOMy_0))*(MIN(abs((float)NOMy_1),abs((float)NOMy_0)));
                denom2 = denom2*denom2;
                T1 = sqrt(denom1 + denom2 + EPS_ROF);
                D1[index] = NOMx_1/T1;
		}		
	}
	
/* differences 2 */
__global__ void D2_func2D_ROF_kernel(float* Input, float* D2, int N, int M)      
    {
		int i1, j1, j2;
		float NOMx_1,NOMy_1,NOMx_0,denom1,denom2,T2;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + N*j;        
        
        if ((i >= 0) && (i < (N)) && (j >= 0) && (j < (M))) {
            
            /* boundary conditions (Neumann reflections) */
                i1 = i + 1; if (i1 >= N) i1 = i-1;
                j1 = j + 1; if (j1 >= M) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1; 
		
                /* Forward-backward differences */
                NOMx_1 = Input[j1*N + i] - Input[index]; /* x+ */
                NOMy_1 = Input[j*N + i1] - Input[index]; /* y+ */
                NOMx_0 = Input[index] - Input[j2*N + i]; /* x- */
                
                denom1 = NOMy_1*NOMy_1;
                denom2 = 0.5f*(signLLT((float)NOMx_1) + signLLT((float)NOMx_0))*(MIN(abs((float)NOMx_1),abs((float)NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrt(denom1 + denom2 + EPS_ROF);
                D2[index] = NOMy_1/T2;	
		}		
	}

 
    /* differences 1 */
__global__ void D1_func3D_ROF_kernel(float* Input, float* D1, int dimX, int dimY, int dimZ)      
    {
		float NOMx_1, NOMy_1, NOMy_0, NOMz_1, NOMz_0, denom1, denom2,denom3, T1;
		int i1,i2,k1,j1,j2,k2;
		
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
      	int index = (dimX*dimY)*k + j*dimX+i;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {
            
                    /* symmetric boundary conditions (Neuman) */
                    i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                    i2 = i - 1; if (i2 < 0) i2 = i+1;
                    j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                    j2 = j - 1; if (j2 < 0) j2 = j+1;
                    k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                    k2 = k - 1; if (k2 < 0) k2 = k+1;
                    
                    /* Forward-backward differences */
                    NOMx_1 = Input[(dimX*dimY)*k + j1*dimX + i] - Input[index]; /* x+ */
                    NOMy_1 = Input[(dimX*dimY)*k + j*dimX + i1] - Input[index]; /* y+ */                    
                    NOMy_0 = Input[index] - Input[(dimX*dimY)*k + j*dimX + i2]; /* y- */
                    
                    NOMz_1 = Input[(dimX*dimY)*k1 + j*dimX + i] - Input[index]; /* z+ */
                    NOMz_0 = Input[index] - Input[(dimX*dimY)*k2 + j*dimX + i]; /* z- */
                    
                    
                    denom1 = NOMx_1*NOMx_1;
                    denom2 = 0.5*(signLLT(NOMy_1) + signLLT(NOMy_0))*(MIN(abs(NOMy_1),abs(NOMy_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(signLLT(NOMz_1) + signLLT(NOMz_0))*(MIN(abs(NOMz_1),abs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T1 = sqrt(denom1 + denom2 + denom3 + EPS_ROF);
                    D1[index] = NOMx_1/T1;	
		}		
	}      

    /* differences 2 */
    __global__ void D2_func3D_ROF_kernel(float* Input, float* D2, int dimX, int dimY, int dimZ)      
    {
		float NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2;
		int i1,i2,k1,j1,j2,k2;
		
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
      	int index = (dimX*dimY)*k + j*dimX+i;     
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {
                    /* symmetric boundary conditions (Neuman) */
                    i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                    i2 = i - 1; if (i2 < 0) i2 = i+1;
                    j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                    j2 = j - 1; if (j2 < 0) j2 = j+1;
                    k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                    k2 = k - 1; if (k2 < 0) k2 = k+1;
                    
                    
                    /* Forward-backward differences */
                    NOMx_1 = Input[(dimX*dimY)*k + (j1)*dimX + i] - Input[index]; /* x+ */
                    NOMy_1 = Input[(dimX*dimY)*k + (j)*dimX + i1] - Input[index]; /* y+ */
                    NOMx_0 = Input[index] - Input[(dimX*dimY)*k + (j2)*dimX + i]; /* x- */
                    NOMz_1 = Input[(dimX*dimY)*k1 + j*dimX + i] - Input[index]; /* z+ */
                    NOMz_0 = Input[index] - Input[(dimX*dimY)*k2 + (j)*dimX + i]; /* z- */
                    
                    
                    denom1 = NOMy_1*NOMy_1;
                    denom2 = 0.5*(signLLT(NOMx_1) + signLLT(NOMx_0))*(MIN(abs(NOMx_1),abs(NOMx_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(signLLT(NOMz_1) + signLLT(NOMz_0))*(MIN(abs(NOMz_1),abs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T2 = sqrt(denom1 + denom2 + denom3 + EPS_ROF);
                    D2[index] = NOMy_1/T2;
		}
	}
	
	  /* differences 3 */
    __global__ void D3_func3D_ROF_kernel(float* Input, float* D3, int dimX, int dimY, int dimZ)      
    {
		float NOMx_1, NOMy_1, NOMx_0, NOMy_0, NOMz_1, denom1, denom2, denom3, T3;
		int i1,i2,k1,j1,j2,k2;
		
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
      	int index = (dimX*dimY)*k + j*dimX+i;     
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {

				i1 = i + 1; if (i1 >= dimX) i1 = i-1;
                i2 = i - 1; if (i2 < 0) i2 = i+1;
                j1 = j + 1; if (j1 >= dimY) j1 = j-1;
                j2 = j - 1; if (j2 < 0) j2 = j+1;
                k1 = k + 1; if (k1 >= dimZ) k1 = k-1;
                k2 = k - 1; if (k2 < 0) k2 = k+1;
                
                /* Forward-backward differences */
                NOMx_1 = Input[(dimX*dimY)*k + (j1)*dimX + i] - Input[index]; /* x+ */
                NOMy_1 = Input[(dimX*dimY)*k + (j)*dimX + i1] - Input[index]; /* y+ */
                NOMy_0 = Input[index] - Input[(dimX*dimY)*k + (j)*dimX + i2]; /* y- */
                NOMx_0 = Input[index] - Input[(dimX*dimY)*k + (j2)*dimX + i]; /* x- */
                NOMz_1 = Input[(dimX*dimY)*k1 + j*dimX + i] - Input[index]; /* z+ */
               
                denom1 = NOMz_1*NOMz_1;
                denom2 = 0.5*(signLLT(NOMx_1) + signLLT(NOMx_0))*(MIN(abs(NOMx_1),abs(NOMx_0)));
                denom2 = denom2*denom2;
                denom3 = 0.5*(signLLT(NOMy_1) + signLLT(NOMy_0))*(MIN(abs(NOMy_1),abs(NOMy_0)));
                denom3 = denom3*denom3;
                T3 = sqrt(denom1 + denom2 + denom3 + EPS_ROF);
                D3[index] = NOMz_1/T3;
		}
	}
/*************************************************************************/
/**********************ROF-LLT-related functions *************************/
/*************************************************************************/

__global__ void Update2D_LLT_ROF_kernel(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D1_ROF, float *D2_ROF, float lambdaROF, float lambdaLLT, float tau, int dimX, int dimY)
{
		
		int i_p, i_m, j_m, j_p;
		float div, laplc, dxx, dyy, dv1, dv2;
	
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
			/* symmetric boundary conditions (Neuman) */
			i_p = i + 1; if (i_p == dimX) i_p = i - 1;
			i_m = i - 1; if (i_m < 0) i_m = i + 1;
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;

			index = j*dimX+i;
					
			/*LLT-related part*/
			dxx = D1_LLT[j*dimX+i_p] - 2.0f*D1_LLT[index] + D1_LLT[j*dimX+i_m];
			dyy = D2_LLT[j_p*dimX+i] - 2.0f*D2_LLT[index] + D2_LLT[j_m*dimX+i];
			laplc = dxx + dyy; /*build Laplacian*/
			/*ROF-related part*/
			dv1 = D1_ROF[index] - D1_ROF[j_m*dimX + i];
            dv2 = D2_ROF[index] - D2_ROF[j*dimX + i_m];
			div = dv1 + dv2; /*build Divirgent*/
            
			/*combine all into one cost function to minimise */
            U[index] += tau*(2.0f*lambdaROF*(div) - lambdaLLT*(laplc) - (U[index] - U0[index]));
		}
}

__global__ void Update3D_LLT_ROF_kernel(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D3_LLT, float *D1_ROF, float *D2_ROF, float *D3_ROF, float lambdaROF, float lambdaLLT, float tau, int dimX, int dimY, int dimZ)
{
	int i_p, i_m, j_m, j_p, k_p, k_m;
	float div, laplc, dxx, dyy, dzz, dv1, dv2, dv3;
	
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {
			
			/* symmetric boundary conditions (Neuman) */
			i_p = i + 1; if (i_p == dimX) i_p = i - 1;
			i_m = i - 1; if (i_m < 0) i_m = i + 1;
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;
			k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
			k_m = k - 1; if (k_m < 0) k_m = k + 1;
        
			int index = (dimX*dimY)*k + j*dimX+i;
      	
			/*LLT-related part*/
			dxx = D1_LLT[(dimX*dimY)*k + j*dimX+i_p] - 2.0f*D1_LLT[index] + D1_LLT[(dimX*dimY)*k + j*dimX+i_m];
			dyy = D2_LLT[(dimX*dimY)*k + j_p*dimX+i] - 2.0f*D2_LLT[index] + D2_LLT[(dimX*dimY)*k + j_m*dimX+i];
			dzz = D3_LLT[(dimX*dimY)*k_p + j*dimX+i] - 2.0f*D3_LLT[index] + D3_LLT[(dimX*dimY)*k_m + j*dimX+i];
			laplc = dxx + dyy + dzz; /*build Laplacian*/
			
			/*ROF-related part*/
			dv1 = D1_ROF[index] - D1_ROF[(dimX*dimY)*k + j_m*dimX+i];
            dv2 = D2_ROF[index] - D2_ROF[(dimX*dimY)*k + j*dimX+i_m];
            dv3 = D3_ROF[index] - D3_ROF[(dimX*dimY)*k_m + j*dimX+i];
			div = dv1 + dv2 + dv3; /*build Divirgent*/
            
			/*combine all into one cost function to minimise */
            U[index] += tau*(2.0f*lambdaROF*(div) - lambdaLLT*(laplc) - (U[index] - U0[index]));
        }
}

/*******************************************************************/
/************************ HOST FUNCTION ****************************/
/*******************************************************************/

extern "C" void LLT_ROF_GPU_main(float *Input, float *Output, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau, int N, int M, int Z)
{
	    // set up device
		int dev = 0;
		int DimTotal;
		DimTotal = N*M*Z;
		CHECK(cudaSetDevice(dev));
        float *d_input, *d_update;
        float *D1_LLT=NULL, *D2_LLT=NULL, *D3_LLT=NULL, *D1_ROF=NULL, *D2_ROF=NULL, *D3_ROF=NULL;
        
	if (Z == 0) {Z = 1;}
	
        CHECK(cudaMalloc((void**)&d_input,DimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_update,DimTotal*sizeof(float)));
        
        CHECK(cudaMalloc((void**)&D1_LLT,DimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&D2_LLT,DimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&D3_LLT,DimTotal*sizeof(float)));
        
        CHECK(cudaMalloc((void**)&D1_ROF,DimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&D2_ROF,DimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&D3_ROF,DimTotal*sizeof(float)));
        
        CHECK(cudaMemcpy(d_input,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_update,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        
    if (Z == 1) {
			// TV - 2D case
            dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
            dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
             
            for(int n=0; n < iterationsNumb; n++) {
                /****************ROF******************/
				/* calculate first-order differences */
                D1_func2D_ROF_kernel<<<dimGrid,dimBlock>>>(d_update, D1_ROF, N, M);
                CHECK(cudaDeviceSynchronize());
				D2_func2D_ROF_kernel<<<dimGrid,dimBlock>>>(d_update, D2_ROF, N, M);
                CHECK(cudaDeviceSynchronize());                
                /****************LLT******************/
                 /* estimate second-order derrivatives */
				der2D_LLT_kernel<<<dimGrid,dimBlock>>>(d_update, D1_LLT, D2_LLT, N, M);
				/* Joint update for ROF and LLT models */
				Update2D_LLT_ROF_kernel<<<dimGrid,dimBlock>>>(d_input, d_update, D1_LLT, D2_LLT, D1_ROF, D2_ROF, lambdaROF, lambdaLLT, tau, N, M);
                CHECK(cudaDeviceSynchronize());
            }
    }
    else {
			// 3D case
            dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
            dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));
           
            for(int n=0; n < iterationsNumb; n++) {
                /****************ROF******************/
				/* calculate first-order differences */
                D1_func3D_ROF_kernel<<<dimGrid,dimBlock>>>(d_update, D1_ROF, N, M, Z);
                CHECK(cudaDeviceSynchronize());
				D2_func3D_ROF_kernel<<<dimGrid,dimBlock>>>(d_update, D2_ROF, N, M, Z);
                CHECK(cudaDeviceSynchronize());        
                D3_func3D_ROF_kernel<<<dimGrid,dimBlock>>>(d_update, D3_ROF, N, M, Z);
                CHECK(cudaDeviceSynchronize());        
                /****************LLT******************/
                 /* estimate second-order derrivatives */
				der3D_LLT_kernel<<<dimGrid,dimBlock>>>(d_update, D1_LLT, D2_LLT, D3_LLT, N, M, Z);
				/* Joint update for ROF and LLT models */
				Update3D_LLT_ROF_kernel<<<dimGrid,dimBlock>>>(d_input, d_update, D1_LLT, D2_LLT, D3_LLT, D1_ROF, D2_ROF, D3_ROF, lambdaROF, lambdaLLT, tau, N, M, Z);
                CHECK(cudaDeviceSynchronize());
            }
    }        
        CHECK(cudaMemcpy(Output,d_update,DimTotal*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_update));
        CHECK(cudaFree(D1_LLT));
        CHECK(cudaFree(D2_LLT));
        CHECK(cudaFree(D3_LLT));
        CHECK(cudaFree(D1_ROF));
        CHECK(cudaFree(D2_ROF));
        CHECK(cudaFree(D3_ROF));
}

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

#include "TV_ROF_GPU_core.h"

/* C-OMP implementation of ROF-TV denoising/regularization model [1] (2D/3D case)
*
* Input Parameters:
* 1. Noisy image/volume [REQUIRED]
* 2. lambda - regularization parameter [REQUIRED]
* 3. tau - marching step for explicit scheme, ~0.1 is recommended [REQUIRED]
* 4. Number of iterations, for explicit scheme >= 150 is recommended [REQUIRED]
*
* Output:
* [1] Regularized image/volume

 * This function is based on the paper by
* [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
*
* D. Kazantsev, 2016-18
*/

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
    
#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8
    
#define BLKXSIZE2D 16
#define BLKYSIZE2D 16
#define EPS 1.0e-5
    
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__host__ __device__ int sign (float x)
{
        return (x > 0) - (x < 0);
}        
   
/*********************2D case****************************/    
    
    /* differences 1 */
    __global__ void D1_func2D(float* Input, float* D1, int N, int M)      
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
                denom2 = 0.5f*(sign((float)NOMy_1) + sign((float)NOMy_0))*(MIN(abs((float)NOMy_1),abs((float)NOMy_0)));
                denom2 = denom2*denom2;
                T1 = sqrt(denom1 + denom2 + EPS);
                D1[index] = NOMx_1/T1;	
		}		
	}       
    
    /* differences 2 */
    __global__ void D2_func2D(float* Input, float* D2, int N, int M)      
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
                denom2 = 0.5f*(sign((float)NOMx_1) + sign((float)NOMx_0))*(MIN(abs((float)NOMx_1),abs((float)NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrt(denom1 + denom2 + EPS);
                D2[index] = NOMy_1/T2;	
		}		
	}
    
    __global__ void TV_kernel2D(float *D1, float *D2, float *Update, float *Input, float lambda, float tau, int N, int M)    
    {
		int i2, j2;
		float dv1,dv2;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + N*j;        
        
        if ((i >= 0) && (i < (N)) && (j >= 0) && (j < (M))) {
            
				/* boundary conditions (Neumann reflections) */                
                i2 = i - 1; if (i2 < 0) i2 = i+1;                
                j2 = j - 1; if (j2 < 0) j2 = j+1; 
                
				/* divergence components  */
                dv1 = D1[index] - D1[j2*N + i];
                dv2 = D2[index] - D2[j*N + i2];                                
                
                Update[index] =  Update[index] + tau*(2.0f*lambda*(dv1 + dv2) - (Update[index] - Input[index]));      
		
		}  
	}   
/*********************3D case****************************/    
 
    /* differences 1 */
    __global__ void D1_func3D(float* Input, float* D1, int dimX, int dimY, int dimZ)      
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
                    denom2 = 0.5*(sign(NOMy_1) + sign(NOMy_0))*(MIN(abs(NOMy_1),abs(NOMy_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(sign(NOMz_1) + sign(NOMz_0))*(MIN(abs(NOMz_1),abs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T1 = sqrt(denom1 + denom2 + denom3 + EPS);
                    D1[index] = NOMx_1/T1;	
		}		
	}      

    /* differences 2 */
    __global__ void D2_func3D(float* Input, float* D2, int dimX, int dimY, int dimZ)      
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
                    denom2 = 0.5*(sign(NOMx_1) + sign(NOMx_0))*(MIN(abs(NOMx_1),abs(NOMx_0)));
                    denom2 = denom2*denom2;
                    denom3 = 0.5*(sign(NOMz_1) + sign(NOMz_0))*(MIN(abs(NOMz_1),abs(NOMz_0)));
                    denom3 = denom3*denom3;
                    T2 = sqrt(denom1 + denom2 + denom3 + EPS);
                    D2[index] = NOMy_1/T2;
		}
	}
	
	  /* differences 3 */
    __global__ void D3_func3D(float* Input, float* D3, int dimX, int dimY, int dimZ)      
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
                denom2 = 0.5*(sign(NOMx_1) + sign(NOMx_0))*(MIN(abs(NOMx_1),abs(NOMx_0)));
                denom2 = denom2*denom2;
                denom3 = 0.5*(sign(NOMy_1) + sign(NOMy_0))*(MIN(abs(NOMy_1),abs(NOMy_0)));
                denom3 = denom3*denom3;
                T3 = sqrt(denom1 + denom2 + denom3 + EPS);
                D3[index] = NOMz_1/T3;		
		}
	}

    __global__ void TV_kernel3D(float *D1, float *D2, float *D3, float *Update, float *Input, float lambda, float tau, int dimX, int dimY, int dimZ)    
    {
		float dv1, dv2, dv3;
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
                    
                    /*divergence components */
                    dv1 = D1[index] - D1[(dimX*dimY)*k + j2*dimX+i];
                    dv2 = D2[index] - D2[(dimX*dimY)*k + j*dimX+i2];
                    dv3 = D3[index] - D3[(dimX*dimY)*k2 + j*dimX+i];
                    
                    Update[index] = Update[index] + tau*(2.0f*lambda*(dv1 + dv2 + dv3) - (Update[index] - Input[index]));
		
		}  
	}

/////////////////////////////////////////////////
// HOST FUNCTION
extern "C" void TV_ROF_GPU(float* Input, float* Output, int N, int M, int Z, int iter, float tau, float lambda)
{
	    // set up device
		int dev = 0;
		CHECK(cudaSetDevice(dev));
		
        float *d_input, *d_update, *d_D1, *d_D2;
        
	if (Z == 0) Z = 1;
        CHECK(cudaMalloc((void**)&d_input,N*M*Z*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_update,N*M*Z*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_D1,N*M*Z*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_D2,N*M*Z*sizeof(float)));
        
        CHECK(cudaMemcpy(d_input,Input,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_update,Input,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));      
        
        if (Z > 1) {
			// TV - 3D case                 
            dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
            dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));            
            
            float *d_D3;
            CHECK(cudaMalloc((void**)&d_D3,N*M*Z*sizeof(float)));
            
            for(int n=0; n < iter; n++) {
                /* calculate differences */
                D1_func3D<<<dimGrid,dimBlock>>>(d_update, d_D1, N, M, Z);				
                CHECK(cudaDeviceSynchronize());
				D2_func3D<<<dimGrid,dimBlock>>>(d_update, d_D2, N, M, Z);				                
                CHECK(cudaDeviceSynchronize());        
                D3_func3D<<<dimGrid,dimBlock>>>(d_update, d_D3, N, M, Z);				
                CHECK(cudaDeviceSynchronize());        
                /*running main kernel*/
                TV_kernel3D<<<dimGrid,dimBlock>>>(d_D1, d_D2, d_D3, d_update, d_input, lambda, tau, N, M, Z);
                CHECK(cudaDeviceSynchronize());
            }
            
            CHECK(cudaFree(d_D3));         
        }
        else {
	    // TV - 2D case
            dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
            dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
             
            for(int n=0; n < iter; n++) {
                /* calculate differences */
                D1_func2D<<<dimGrid,dimBlock>>>(d_update, d_D1, N, M);				
                CHECK(cudaDeviceSynchronize());
				D2_func2D<<<dimGrid,dimBlock>>>(d_update, d_D2, N, M);				
                CHECK(cudaDeviceSynchronize());        
                /*running main kernel*/
                TV_kernel2D<<<dimGrid,dimBlock>>>(d_D1, d_D2, d_update, d_input, lambda, tau, N, M);
                CHECK(cudaDeviceSynchronize());
            }
        }        
        CHECK(cudaMemcpy(Output,d_update,N*M*Z*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_update));
        CHECK(cudaFree(d_D1));
        CHECK(cudaFree(d_D2));        
        cudaDeviceReset(); 
}

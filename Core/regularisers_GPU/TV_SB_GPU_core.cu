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

#include "TV_SB_GPU_core.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

/* CUDA implementation of Split Bregman - TV denoising-regularisation model (2D/3D) [1]
*
* Input Parameters:
* 1. Noisy image/volume
* 2. lambda - regularisation parameter
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon - tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
* 6. nonneg: 'nonnegativity (0 is OFF by default) [OPTIONAL parameter]
* 7. print information: 0 (off) or 1 (on)  [OPTIONAL parameter]
*
* Output:
* 1. Filtered/regularized image
*
* [1]. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.
*/

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
struct square { __host__ __device__ float operator()(float x) { return x * x; } };

/************************************************/
/*****************2D modules*********************/
/************************************************/
__global__ void gauss_seidel2D_kernel(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Bx, float *By, float lambda, float mu, float normConst, int N, int M, int ImSize)
{
    
    float sum;
    int i1,i2,j1,j2;
     
    //calculate each thread global index
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = j*N+i;
    
    if ((i < N) && (j < M)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        j2 = j-1; if (j2 < 0) j2 = j+1;
        
        sum = Dx[j*N+i2] - Dx[index] + Dy[j2*N+i] - Dy[index] - Bx[j*N+i2] + Bx[index] - By[j2*N+i] + By[index];
        sum += U_prev[j*N+i1] + U_prev[j*N+i2] + U_prev[j1*N+i] + U_prev[j2*N+i];
        sum *= lambda;
        sum += mu*A[index];
        U[index] = normConst*sum; //Write final result to global memory
    }
    return;
}
__global__ void updDxDy_shrinkAniso2D_kernel(float *U, float *Dx, float *Dy, float *Bx, float *By, float lambda, int N, int M, int ImSize)
{
    
    int i1,j1;
    float val1, val11, val2, val22, denom_lam;
    denom_lam = 1.0f/lambda;
     
    //calculate each thread global index
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = j*N+i;
    
    if ((i < N) && (j < M)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        j1 = j+1; if (j1 == M) j1 = j-1;
                
            val1 = (U[j*N+i1] - U[index]) + Bx[index];
            val2 = (U[j1*N+i] - U[index]) + By[index];
            
            val11 = abs(val1) - denom_lam; if (val11 < 0) val11 = 0;
            val22 = abs(val2) - denom_lam; if (val22 < 0) val22 = 0;
            
            if (val1 !=0) Dx[index] = (val1/abs(val1))*val11; else Dx[index] = 0;
            if (val2 !=0) Dy[index] = (val2/abs(val2))*val22; else Dy[index] = 0;
    }
    return;
}

__global__ void updDxDy_shrinkIso2D_kernel(float *U, float *Dx, float *Dy, float *Bx, float *By, float lambda, int N, int M, int ImSize)
{
    
    int i1,j1;
    float val1, val11, val2, denom_lam, denom;
    denom_lam = 1.0f/lambda;
     
    //calculate each thread global index
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = j*N+i;
    
    if ((i < N) && (j < M)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        
            val1 = (U[j*N+i1] - U[index]) + Bx[index];
            val2 = (U[j1*N+i] - U[index]) + By[index];
            
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
    }
    return;
}

__global__ void updBxBy2D_kernel(float *U, float *Dx, float *Dy, float *Bx, float *By, int N, int M, int ImSize)
{    
    int i1,j1;
     
    //calculate each thread global index
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = j*N+i;
    
    if ((i < N) && (j < M)) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == N) i1 = i-1;
            j1 = j+1; if (j1 == M) j1 = j-1;
            
            Bx[index] += (U[j*N+i1] - U[index]) - Dx[index];
            By[index] += (U[j1*N+i] - U[index]) - Dy[index];
    }
    return;
}


/************************************************/
/*****************3D modules*********************/
/************************************************/
__global__ void gauss_seidel3D_kernel(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, float lambda, float mu, float normConst, int N, int M, int Z, int ImSize)
{
    
    float sum,d_val,b_val;
    int i1,i2,j1,j2,k1,k2;
     
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k < Z)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        j2 = j-1; if (j2 < 0) j2 = j+1;
        k1 = k+1; if (k1 == Z) k1 = k-1;
        k2 = k-1; if (k2 < 0) k2 = k+1;
        
        d_val = Dx[(N*M)*k + j*N+i2] - Dx[index] + Dy[(N*M)*k + j2*N+i] - Dy[index] + Dz[(N*M)*k2 + j*N+i] - Dz[index];
        b_val = -Bx[(N*M)*k + j*N+i2] + Bx[index] - By[(N*M)*k + j2*N+i] + By[index] - Bz[(N*M)*k2 + j*N+i] + Bz[index];
        sum = d_val + b_val;
        sum += U_prev[(N*M)*k + j*N+i1] + U_prev[(N*M)*k + j*N+i2] + U_prev[(N*M)*k + j1*N+i] + U_prev[(N*M)*k + j2*N+i] + U_prev[(N*M)*k1 + j*N+i] + U_prev[(N*M)*k2 + j*N+i];
        sum *= lambda;
        sum += mu*A[index];
        U[index] = normConst*sum;
    }
    return;
}
__global__ void updDxDy_shrinkAniso3D_kernel(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, float lambda, int N, int M, int Z, int ImSize)
{
    
    int i1,j1,k1;
    float val1, val11, val2, val3, val22, val33, denom_lam;
    denom_lam = 1.0f/lambda;
     
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k < Z)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        k1 = k+1; if (k1 == Z) k1 = k-1;
                
            val1 = (U[(N*M)*k + i1 + N*j] - U[index]) + Bx[index];
            val2 = (U[(N*M)*k + i + N*j1] - U[index]) + By[index];
            val3 = (U[(N*M)*k1 + i + N*j] - U[index]) + Bz[index];
            
            val11 = abs(val1) - denom_lam; if (val11 < 0.0f) val11 = 0.0f;
            val22 = abs(val2) - denom_lam; if (val22 < 0.0f) val22 = 0.0f;
            val33 = abs(val3) - denom_lam; if (val33 < 0.0f) val33 = 0.0f;
            
            if (val1 !=0.0f) Dx[index] = (val1/abs(val1))*val11; else Dx[index] = 0.0f;
            if (val2 !=0.0f) Dy[index] = (val2/abs(val2))*val22; else Dy[index] = 0.0f;
            if (val3 !=0.0f) Dz[index] = (val3/abs(val3))*val33; else Dz[index] = 0.0f;
    }
    return;
}

__global__ void updDxDy_shrinkIso3D_kernel(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, float lambda, int N, int M, int Z, int ImSize)
{
    
    int i1,j1,k1;
    float val1, val11, val2, val3, denom_lam, denom;
    denom_lam = 1.0f/lambda;
     
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k < Z)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        k1 = k+1; if (k1 == Z) k1 = k-1;
        
            val1 = (U[(N*M)*k + i1 + N*j] - U[index]) + Bx[index];
            val2 = (U[(N*M)*k + i + N*j1] - U[index]) + By[index];
            val3 = (U[(N*M)*k1 + i + N*j] - U[index]) + Bz[index];
            
            denom = sqrt(val1*val1 + val2*val2 + val3*val3);
            
            val11 = (denom - denom_lam); if (val11 < 0.0f) val11 = 0.0f;
            
            if (denom != 0.0f) {
                Dx[index] = val11*(val1/denom);
                Dy[index] = val11*(val2/denom);
                Dz[index] = val11*(val3/denom);
            }
            else {
                Dx[index] = 0.0f;
                Dy[index] = 0.0f;
                Dz[index] = 0.0f;
            }
    }
    return;
}

__global__ void updBxBy3D_kernel(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int N, int M, int Z, int ImSize)
{    
    int i1,j1,k1;
     
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k < Z)) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == N) i1 = i-1;
            j1 = j+1; if (j1 == M) j1 = j-1;
            k1 = k+1; if (k1 == Z) k1 = k-1;
            
            Bx[index] += (U[(N*M)*k + i1 + N*j] - U[index]) - Dx[index];
            By[index] += (U[(N*M)*k + i + N*j1] - U[index]) - Dy[index];
            Bz[index] += (U[(N*M)*k1 + i + N*j] - U[index]) - Dz[index];
    }
    return;
}

__global__ void SBcopy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

__global__ void SBcopy_kernel3D(float *Input, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

__global__ void SBResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

__global__ void SBResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/********************* MAIN HOST FUNCTION ******************/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
extern "C" void TV_SB_GPU_main(float *Input, float *Output, float mu, int iter, float epsil, int methodTV, int printM, int dimX, int dimY, int dimZ)
{
    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }
    
	int ll, DimTotal;
	float re, lambda, normConst;
    int count = 0;
    mu = 1.0f/mu;
	lambda = 2.0f*mu;

    if (dimZ <= 1) {
		/*2D verson*/
		DimTotal = dimX*dimY;
		normConst = 1.0f/(mu + 4.0f*lambda);
		float *d_input, *d_update, *d_res, *d_update_prev=NULL, *Dx=NULL, *Dy=NULL, *Bx=NULL, *By=NULL;
   
		dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
		dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));
    
		/*allocate space for images on device*/
		checkCudaErrors( cudaMalloc((void**)&d_input,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&d_update,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&d_update_prev,DimTotal*sizeof(float)) );
		if (epsil != 0.0f) checkCudaErrors( cudaMalloc((void**)&d_res,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Dx,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Dy,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Bx,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&By,DimTotal*sizeof(float)) );
    
        checkCudaErrors( cudaMemcpy(d_input,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors( cudaMemcpy(d_update,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        cudaMemset(Dx, 0, DimTotal*sizeof(float));
        cudaMemset(Dy, 0, DimTotal*sizeof(float));
        cudaMemset(Bx, 0, DimTotal*sizeof(float));
        cudaMemset(By, 0, DimTotal*sizeof(float));

        /********************** Run CUDA 2D kernels here ********************/   
        /* The main kernel */
        for (ll = 0; ll < iter; ll++) {
        
        /* storing old value */
        SBcopy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );  

		 /* perform two GS iterations (normally 2 is enough for the convergence) */
        gauss_seidel2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Bx, By, lambda, mu, normConst, dimX, dimY, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        SBcopy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );  
        /* 2nd GS iteration */
        gauss_seidel2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Bx, By, lambda, mu, normConst, dimX, dimY, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        
        /* TV-related step */
          if (methodTV == 1)  updDxDy_shrinkAniso2D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Bx, By, lambda, dimX, dimY, DimTotal);
          else updDxDy_shrinkIso2D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Bx, By, lambda, dimX, dimY, DimTotal);
            
        /* update for Bregman variables */
        updBxBy2D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Bx, By, dimX, dimY, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        
          if (epsil != 0.0f) {
                /* calculate norm - stopping rules using the Thrust library */
                SBResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_update_prev, d_res, dimX, dimY, DimTotal);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );               
                
                thrust::device_vector<float> d_vec(d_res, d_res + DimTotal);
                float reduction = sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), square(), 0.0f, thrust::plus<float>()));		
                thrust::device_vector<float> d_vec2(d_update, d_update + DimTotal);  		
                float reduction2 = sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), square(), 0.0f, thrust::plus<float>()));
                    
                re = (reduction/reduction2);      
                if (re < epsil)  count++;
                    if (count > 4) break;
          }
        
        }
        if (printM == 1) printf("SB-TV iterations stopped at iteration %i \n", ll);   
            /***************************************************************/    
            //copy result matrix from device to host memory
            cudaMemcpy(Output,d_update,DimTotal*sizeof(float),cudaMemcpyDeviceToHost);
    
            cudaFree(d_input);
            cudaFree(d_update);
            cudaFree(d_update_prev);
            if (epsil != 0.0f) cudaFree(d_res);
            cudaFree(Dx);
            cudaFree(Dy);
            cudaFree(Bx);
            cudaFree(By);
    }
    else {
		/*3D verson*/
		DimTotal = dimX*dimY*dimZ;
		normConst = 1.0f/(mu + 6.0f*lambda);
		float *d_input, *d_update, *d_res, *d_update_prev=NULL, *Dx=NULL, *Dy=NULL, *Dz=NULL, *Bx=NULL, *By=NULL, *Bz=NULL;
   
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKZSIZE));
    
		/*allocate space for images on device*/
		checkCudaErrors( cudaMalloc((void**)&d_input,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&d_update,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&d_update_prev,DimTotal*sizeof(float)) );
		if (epsil != 0.0f) checkCudaErrors( cudaMalloc((void**)&d_res,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Dx,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Dy,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Dz,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Bx,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&By,DimTotal*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&Bz,DimTotal*sizeof(float)) );
    
        checkCudaErrors( cudaMemcpy(d_input,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors( cudaMemcpy(d_update,Input,DimTotal*sizeof(float),cudaMemcpyHostToDevice));
        cudaMemset(Dx, 0, DimTotal*sizeof(float));
        cudaMemset(Dy, 0, DimTotal*sizeof(float));
        cudaMemset(Dz, 0, DimTotal*sizeof(float));
        cudaMemset(Bx, 0, DimTotal*sizeof(float));
        cudaMemset(By, 0, DimTotal*sizeof(float));
        cudaMemset(Bz, 0, DimTotal*sizeof(float));

        /********************** Run CUDA 3D kernels here ********************/   
        /* The main kernel */
        for (ll = 0; ll < iter; ll++) {
        
        /* storing old value */
        SBcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, dimZ, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );

		 /* perform two GS iterations (normally 2 is enough for the convergence) */
        gauss_seidel3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Dz, Bx, By, Bz, lambda, mu, normConst, dimX, dimY, dimZ, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        SBcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, dimZ, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );  
        /* 2nd GS iteration */
        gauss_seidel3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Dz, Bx, By, Bz, lambda, mu, normConst, dimX, dimY, dimZ, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        
        /* TV-related step */
          if (methodTV == 1)  updDxDy_shrinkAniso3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, lambda, dimX, dimY, dimZ, DimTotal);
          else updDxDy_shrinkIso3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, lambda, dimX, dimY, dimZ, DimTotal);
            
        /* update for Bregman variables */
        updBxBy3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, DimTotal);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() ); 
        
          if (epsil != 0.0f) {
                /* calculate norm - stopping rules using the Thrust library */
                SBResidCalc3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_update_prev, d_res, dimX, dimY, dimZ, DimTotal);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );               
                
                thrust::device_vector<float> d_vec(d_res, d_res + DimTotal);
                float reduction = sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), square(), 0.0f, thrust::plus<float>()));		
                thrust::device_vector<float> d_vec2(d_update, d_update + DimTotal);  		
                float reduction2 = sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), square(), 0.0f, thrust::plus<float>()));
                    
                re = (reduction/reduction2);
                if (re < epsil)  count++;
                    if (count > 4) break;
          }
        }
        if (printM == 1) printf("SB-TV iterations stopped at iteration %i \n", ll);   
            /***************************************************************/    
            //copy result matrix from device to host memory
            cudaMemcpy(Output,d_update,DimTotal*sizeof(float),cudaMemcpyDeviceToHost);
    
            cudaFree(d_input);
            cudaFree(d_update);
            cudaFree(d_update_prev);
            if (epsil != 0.0f) cudaFree(d_res);
            cudaFree(Dx);
            cudaFree(Dy);
            cudaFree(Dz);
            cudaFree(Bx);
            cudaFree(By);
            cudaFree(Bz);
    } 
    //cudaDeviceReset(); 
}

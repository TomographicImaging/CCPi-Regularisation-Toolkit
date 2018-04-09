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

#include "TV_FGP_GPU_core.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

/* CUDA implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume 
 * 2. lambdaPar - regularization parameter 
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant 
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default) 
 * 7. print information: 0 (off) or 1 (on) 
 *
 * Output:
 * [1] Filtered/regularized image
 *
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
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
__global__ void Obj_func2D_kernel(float *Ad, float *D, float *R1, float *R2, int N, int M, int ImSize, float lambda)
{
    
    float val1,val2;
    
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = xIndex + N*yIndex; 
    
    if ((xIndex < N) && (yIndex < M)) {        
        if (xIndex <= 0) {val1 = 0.0f;} else {val1 = R1[(xIndex-1) + N*yIndex];}
        if (yIndex <= 0) {val2 = 0.0f;} else {val2 = R2[xIndex + N*(yIndex-1)];}
        //Write final result to global memory
        D[index] = Ad[index] - lambda*(R1[index] + R2[index] - val1 - val2);
    }
    return;
}

__global__ void Grad_func2D_kernel(float *P1, float *P2, float *D, float *R1, float *R2, int N, int M, int ImSize, float multip)
{
    
    float val1,val2;
    
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if ((xIndex < N) && (yIndex < M)) {        
        
        /* boundary conditions */
        if (xIndex >= N-1) val1 = 0.0f; else val1 = D[index] - D[(xIndex+1) + N*yIndex];
        if (yIndex >= M-1) val2 = 0.0f; else val2 = D[index] - D[(xIndex) + N*(yIndex + 1)];
        
        //Write final result to global memory
        P1[index] = R1[index] + multip*val1;
        P2[index] = R2[index] + multip*val2;
    }
    return;
}

__global__ void Proj_func2D_iso_kernel(float *P1, float *P2, int N, int M, int ImSize)
{
    
    float denom;    
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if ((xIndex < N) && (yIndex < M)) { 
        denom = pow(P1[index],2) +  pow(P2[index],2);        
        if (denom > 1.0f) {
            P1[index] = P1[index]/sqrt(denom);
            P2[index] = P2[index]/sqrt(denom);
        }
    }
    return;
}
__global__ void Proj_func2D_aniso_kernel(float *P1, float *P2, int N, int M, int ImSize)
{
    
    float val1, val2;    
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if ((xIndex < N) && (yIndex < M)) { 
                val1 = abs(P1[index]);
                val2 = abs(P2[index]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                P1[index] = P1[index]/val1;
                P2[index] = P2[index]/val2;
    }
    return;
}
__global__ void Rupd_func2D_kernel(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, float multip2, int N, int M, int ImSize)
{
    //calculate each thread global index
    const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
    const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if ((xIndex < N) && (yIndex < M)) { 
        R1[index] = P1[index] + multip2*(P1[index] - P1_old[index]);
        R2[index] = P2[index] + multip2*(P2[index] - P2_old[index]);
    }
    return;
}
__global__ void nonneg2D_kernel(float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        if (Output[index] < 0.0f) Output[index] = 0.0f;
    }
}
__global__ void copy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input[index];
    }
}
__global__ void ResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}   
/************************************************/
/*****************3D modules*********************/
/************************************************/
__global__ void Obj_func3D_kernel(float *Ad, float *D, float *R1, float *R2, float *R3, int N, int M, int Z, int ImSize, float lambda)
{
    
    float val1,val2,val3;
    
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k < Z)) {      
        if (i <= 0) {val1 = 0.0f;} else {val1 = R1[(N*M)*(k) + (i-1) + N*j];}
        if (j <= 0) {val2 = 0.0f;} else {val2 = R2[(N*M)*(k) + i + N*(j-1)];}
        if (k <= 0) {val3 = 0.0f;} else {val3 = R3[(N*M)*(k-1) + i + N*j];}
        //Write final result to global memory
        D[index] = Ad[index] - lambda*(R1[index] + R2[index] + R3[index] - val1 - val2 - val3);
    }
    return;
}

__global__ void Grad_func3D_kernel(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, int N, int M, int Z, int ImSize, float multip)
{
    
    float val1,val2,val3;
    
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k <  Z)) {       
        /* boundary conditions */
        if (i >= N-1) val1 = 0.0f; else val1 = D[index] - D[(N*M)*(k) + (i+1) + N*j];
        if (j >= M-1) val2 = 0.0f; else val2 = D[index] - D[(N*M)*(k) + i + N*(j+1)];
        if (k >= Z-1) val3 = 0.0f; else val3 = D[index] - D[(N*M)*(k+1) + i + N*j];
        
        //Write final result to global memory
        P1[index] = R1[index] + multip*val1;
        P2[index] = R2[index] + multip*val2;
        P3[index] = R3[index] + multip*val3;
    }
    return;
}

__global__ void Proj_func3D_iso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
{
    
    float denom,sq_denom;    
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k <  Z)) {
        denom = pow(P1[index],2) +  pow(P2[index],2) + pow(P3[index],2);
        
        if (denom > 1.0f) {
            sq_denom = 1.0f/sqrt(denom);
            P1[index] = P1[index]*sq_denom;
            P2[index] = P2[index]*sq_denom;
            P3[index] = P3[index]*sq_denom;
        }
    }
    return;
}

__global__ void Proj_func3D_aniso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
{
    
    float val1, val2, val3;    
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k <  Z)) {
                val1 = abs(P1[index]);
                val2 = abs(P2[index]);
                val3 = abs(P3[index]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                if (val3 < 1.0f) {val3 = 1.0f;}
                P1[index] = P1[index]/val1;
                P2[index] = P2[index]/val2;
                P3[index] = P3[index]/val3;
    }
    return;
}


__global__ void Rupd_func3D_kernel(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, float multip2, int N, int M, int Z, int ImSize)
{
    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if ((i < N) && (j < M) && (k <  Z)) { 
        R1[index] = P1[index] + multip2*(P1[index] - P1_old[index]);
        R2[index] = P2[index] + multip2*(P2[index] - P2_old[index]);
        R3[index] = P3[index] + multip2*(P3[index] - P3_old[index]);
    }
    return;
}

__global__ void nonneg3D_kernel(float* Output, int N, int M, int Z, int num_total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if (index < num_total)	{
        if (Output[index] < 0.0f) Output[index] = 0.0f;
    }
}

__global__ void copy_kernel3D(float *Input, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if (index < num_total)	{
        Output[index] = Input[index];
    }
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

////////////MAIN HOST FUNCTION ///////////////
extern "C" void TV_FGP_GPU_main(float *Input, float *Output, float lambdaPar, int iter, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ)
{
    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }
    
    int count = 0, i;
    float re, multip,multip2;    
	float tk = 1.0f;
    float tkp1=1.0f;
        
    if (dimZ <= 1) {
		/*2D verson*/
		int ImSize = dimX*dimY;    
		float *d_input, *d_update=NULL, *d_update_prev=NULL, *P1=NULL, *P2=NULL, *P1_prev=NULL, *P2_prev=NULL, *R1=NULL, *R2=NULL;
   
		dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
		dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));
    
		/*allocate space for images on device*/
		checkCudaErrors( cudaMalloc((void**)&d_input,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&d_update,ImSize*sizeof(float)) );
		if (epsil != 0.0f) checkCudaErrors( cudaMalloc((void**)&d_update_prev,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&P1,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&P2,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&P1_prev,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&P2_prev,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&R1,ImSize*sizeof(float)) );
		checkCudaErrors( cudaMalloc((void**)&R2,ImSize*sizeof(float)) );
    
        checkCudaErrors( cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        cudaMemset(P1, 0, ImSize*sizeof(float));
        cudaMemset(P2, 0, ImSize*sizeof(float));
        cudaMemset(P1_prev, 0, ImSize*sizeof(float));
        cudaMemset(P2_prev, 0, ImSize*sizeof(float));
        cudaMemset(R1, 0, ImSize*sizeof(float));
        cudaMemset(R2, 0, ImSize*sizeof(float));

        /********************** Run CUDA 2D kernel here ********************/    
        multip = (1.0f/(8.0f*lambdaPar));
    
        /* The main kernel */
        for (i = 0; i < iter; i++) {
        
            /* computing the gradient of the objective function */
            Obj_func2D_kernel<<<dimGrid,dimBlock>>>(d_input, d_update, R1, R2, dimX, dimY, ImSize, lambdaPar);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
            
            if (nonneg != 0) {
            nonneg2D_kernel<<<dimGrid,dimBlock>>>(d_update, dimX, dimY, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() ); }
                    
            /*Taking a step towards minus of the gradient*/
            Grad_func2D_kernel<<<dimGrid,dimBlock>>>(P1, P2, d_update, R1, R2, dimX, dimY, ImSize, multip);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            /* projection step */
            if (methodTV == 0) Proj_func2D_iso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, ImSize); /*isotropic TV*/
            else Proj_func2D_aniso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, ImSize); /*anisotropic TV*/            
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            multip2 = ((tk-1.0f)/tkp1);
        
            Rupd_func2D_kernel<<<dimGrid,dimBlock>>>(P1, P1_prev, P2, P2_prev, R1, R2, tkp1, tk, multip2, dimX, dimY, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            if (epsil != 0.0f) {
                /* calculate norm - stopping rules using the Thrust library */					
                ResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_update_prev, P1_prev, dimX, dimY, ImSize);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );               
                
                thrust::device_vector<float> d_vec(P1_prev, P1_prev + ImSize);  		
                float reduction = sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), square(), 0.0f, thrust::plus<float>()));		
                thrust::device_vector<float> d_vec2(d_update, d_update + ImSize);  		
                float reduction2 = sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), square(), 0.0f, thrust::plus<float>()));
                    
                re = (reduction/reduction2);      
                if (re < epsil)  count++;
                    if (count > 4) break;       
             
                copy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, ImSize);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors(cudaPeekAtLastError() );                                              
            }                  
        
            copy_kernel2D<<<dimGrid,dimBlock>>>(P1, P1_prev, dimX, dimY, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            copy_kernel2D<<<dimGrid,dimBlock>>>(P2, P2_prev, dimX, dimY, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );       
 
            tk = tkp1;
        }
        if (printM == 1) printf("FGP-TV iterations stopped at iteration %i \n", i);   
            /***************************************************************/    
            //copy result matrix from device to host memory
            cudaMemcpy(Output,d_update,ImSize*sizeof(float),cudaMemcpyDeviceToHost);
    
            cudaFree(d_input);
            cudaFree(d_update);
            if (epsil != 0.0f) cudaFree(d_update_prev);
            cudaFree(P1);
            cudaFree(P2);
            cudaFree(P1_prev);
            cudaFree(P2_prev);
            cudaFree(R1);
            cudaFree(R2);
    }
    else {
            /*3D verson*/
            int ImSize = dimX*dimY*dimZ;    
            float *d_input, *d_update=NULL, *P1=NULL, *P2=NULL, *P3=NULL, *P1_prev=NULL, *P2_prev=NULL, *P3_prev=NULL, *R1=NULL, *R2=NULL, *R3=NULL;
   
            dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
            dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKZSIZE));
    
            /*allocate space for images on device*/
            checkCudaErrors( cudaMalloc((void**)&d_input,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&d_update,ImSize*sizeof(float)) );            
            checkCudaErrors( cudaMalloc((void**)&P1,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&P2,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&P3,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&P1_prev,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&P2_prev,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&P3_prev,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&R1,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&R2,ImSize*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&R3,ImSize*sizeof(float)) );
    
            checkCudaErrors( cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            cudaMemset(P1, 0, ImSize*sizeof(float));
            cudaMemset(P2, 0, ImSize*sizeof(float));
            cudaMemset(P3, 0, ImSize*sizeof(float));
            cudaMemset(P1_prev, 0, ImSize*sizeof(float));
            cudaMemset(P2_prev, 0, ImSize*sizeof(float));
            cudaMemset(P3_prev, 0, ImSize*sizeof(float));
            cudaMemset(R1, 0, ImSize*sizeof(float));
            cudaMemset(R2, 0, ImSize*sizeof(float));
            cudaMemset(R3, 0, ImSize*sizeof(float));
            /********************** Run CUDA 3D kernel here ********************/    
            multip = (1.0f/(26.0f*lambdaPar));
    
            /* The main kernel */
        for (i = 0; i < iter; i++) {
        
            /* computing the gradient of the objective function */
            Obj_func3D_kernel<<<dimGrid,dimBlock>>>(d_input, d_update, R1, R2, R3, dimX, dimY, dimZ, ImSize, lambdaPar);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            if (nonneg != 0) {
            nonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update, dimX, dimY, dimZ, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() ); }
            
            /*Taking a step towards minus of the gradient*/
            Grad_func3D_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, d_update, R1, R2, R3, dimX, dimY, dimZ, ImSize, multip);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            /* projection step */
            if (methodTV == 0) Proj_func3D_iso_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, dimX, dimY, dimZ, ImSize); /* isotropic kernel */
            else Proj_func3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, dimX, dimY, dimZ, ImSize); /* anisotropic kernel */
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            multip2 = ((tk-1.0f)/tkp1);
        
            Rupd_func3D_kernel<<<dimGrid,dimBlock>>>(P1, P1_prev, P2, P2_prev, P3, P3_prev, R1, R2, R3, tkp1, tk, multip2, dimX, dimY, dimZ, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );           
        
            copy_kernel3D<<<dimGrid,dimBlock>>>(P1, P1_prev, dimX, dimY, dimZ, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );
        
            copy_kernel3D<<<dimGrid,dimBlock>>>(P2, P2_prev, dimX, dimY, dimZ, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );   
            
            copy_kernel3D<<<dimGrid,dimBlock>>>(P3, P3_prev, dimX, dimY, dimZ, ImSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );      
 
            tk = tkp1;
        }
        if (printM == 1) printf("FGP-TV iterations stopped at iteration %i \n", i);   
            /***************************************************************/    
            //copy result matrix from device to host memory
            cudaMemcpy(Output,d_update,ImSize*sizeof(float),cudaMemcpyDeviceToHost);
    
            cudaFree(d_input);
            cudaFree(d_update);            
            cudaFree(P1);
            cudaFree(P2);
            cudaFree(P3);
            cudaFree(P1_prev);
            cudaFree(P2_prev);
            cudaFree(P3_prev);
            cudaFree(R1);
            cudaFree(R2);        
            cudaFree(R3);        
    } 
    cudaDeviceReset(); 
}

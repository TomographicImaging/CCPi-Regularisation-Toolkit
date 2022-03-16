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

#include "TV_PD_GPU_core.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

/* CUDA implementation of Primal-Dual TV [1] by Chambolle Pock denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. lipschitz_const: convergence related parameter
 * 6. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 7. nonneg: 'nonnegativity (0 is OFF by default, 1 is ON)

 * Output:
 * [1] TV - Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * [1] Antonin Chambolle, Thomas Pock. "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", 2010
 */

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
// struct square { __host__ __device__ float operator()(float x) { return x * x; } };

/************************************************/
/*****************2D modules*********************/
/************************************************/

__global__ void dualPD_kernel(float *U, float *P1, float *P2, float sigma, int N, int M)
{

   //calculate each thread global index
   const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
   const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;

   int index = xIndex + N*yIndex;

   if ((xIndex < N) && (yIndex < M)) {
     if (xIndex == N-1) P1[index] += sigma*(U[(xIndex-1) + N*yIndex] - U[index]);
     else P1[index] += sigma*(U[(xIndex+1) + N*yIndex] - U[index]);
     if (yIndex == M-1) P2[index] += sigma*(U[xIndex + N*(yIndex-1)] - U[index]);
     else  P2[index] += sigma*(U[xIndex + N*(yIndex+1)] - U[index]);
   }
   return;
}
__global__ void Proj_funcPD2D_iso_kernel(float *P1, float *P2, int N, int M, int ImSize)
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
__global__ void Proj_funcPD2D_aniso_kernel(float *P1, float *P2, int N, int M, int ImSize)
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
__global__ void DivProj2D_kernel(float *U, float *Input, float *P1, float *P2, float lt, float tau, int N, int M)
{
   float P_v1, P_v2, div_var;

   //calculate each thread global index
   const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
   const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;

   int index = xIndex + N*yIndex;

   if ((xIndex < N) && (yIndex < M)) {
     if (xIndex == 0) P_v1 = -P1[index];
     else P_v1 = -(P1[index] - P1[(xIndex-1) + N*yIndex]);
     if (yIndex == 0) P_v2 = -P2[index];
     else  P_v2 = -(P2[index] - P2[xIndex + N*(yIndex-1)]);
     div_var = P_v1 + P_v2;
     U[index] = (U[index] - tau*div_var + lt*Input[index])/(1.0 + lt);
   }
   return;
}
__global__ void PDnonneg2D_kernel(float* Output, int N, int M, int num_total)
{
   int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   int index = xIndex + N*yIndex;

   if (index < num_total)	{
       if (Output[index] < 0.0f) Output[index] = 0.0f;
   }
}
/************************************************/
/*****************3D modules*********************/
/************************************************/
__global__ void dualPD3D_kernel(float *U, float *P1, float *P2, float *P3, float sigma, int N, int M, int Z)
{

  //calculate each thread global index
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int index = (N*M)*k + i + N*j;

  if ((i < N) && (j < M) && (k < Z)) {
     if (i == N-1) P1[index] += sigma*(U[(N*M)*k + (i-1) + N*j] - U[index]);
     else P1[index] += sigma*(U[(N*M)*k + (i+1) + N*j] - U[index]);
     if (j == M-1) P2[index] += sigma*(U[(N*M)*k + i + N*(j-1)] - U[index]);
     else  P2[index] += sigma*(U[(N*M)*k + i + N*(j+1)] - U[index]);
     if (k == Z-1) P3[index] += sigma*(U[(N*M)*(k-1) + i + N*j] - U[index]);
     else  P3[index] += sigma*(U[(N*M)*(k+1) + i + N*j] - U[index]);
   }
   return;
}
__global__ void Proj_funcPD3D_iso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
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
           P1[index] *= sq_denom;
           P2[index] *= sq_denom;
           P3[index] *= sq_denom;
       }
   }
   return;
}
__global__ void Proj_funcPD3D_aniso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
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
               P1[index] /= val1;
               P2[index] /= val2;
               P3[index] /= val3;
   }
   return;
}
__global__ void DivProj3D_kernel(float *U, float *Input, float *P1, float *P2, float *P3, float lt, float tau, int N, int M, int Z)
{
   float P_v1, P_v2, P_v3, div_var;

   //calculate each thread global index
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   int k = blockDim.z * blockIdx.z + threadIdx.z;

   int index = (N*M)*k + i + N*j;

   if ((i < N) && (j < M) && (k <  Z)) {
     if (i == 0) P_v1 = -P1[index];
     else P_v1 = -(P1[index] - P1[(N*M)*k + (i-1) + N*j]);
     if (j == 0) P_v2 = -P2[index];
     else  P_v2 = -(P2[index] - P2[(N*M)*k + i + N*(j-1)]);
     if (k == 0) P_v3 = -P3[index];
     else  P_v3 = -(P3[index] - P3[(N*M)*(k-1) + i + N*j]);
     div_var = P_v1 + P_v2 + P_v3;
     U[index] = (U[index] - tau*div_var + lt*Input[index])/(1.0 + lt);
   }
   return;
}

__global__ void PDnonneg3D_kernel(float* Output, int N, int M, int Z, int num_total)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   int k = blockDim.z * blockIdx.z + threadIdx.z;

   int index = (N*M)*k + i + N*j;

   if (index < num_total)	{
       if (Output[index] < 0.0f) Output[index] = 0.0f;
   }
}
__global__ void PDcopy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
   int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   int index = xIndex + N*yIndex;

   if (index < num_total)	{
       Output[index] = Input[index];
   }
}

__global__ void PDcopy_kernel3D(float *Input, float* Output, int N, int M, int Z, int num_total)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   int k = blockDim.z * blockIdx.z + threadIdx.z;

   int index = (N*M)*k + i + N*j;

   if (index < num_total)	{
       Output[index] = Input[index];
   }
}

__global__ void getU2D_kernel(float *Input, float *Input_old, float theta, int N, int M, int num_total)
{
   int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   int index = xIndex + N*yIndex;

   if (index < num_total)	{
       Input[index] += theta*(Input[index] - Input_old[index]);
   }
}

__global__ void getU3D_kernel(float *Input, float *Input_old, float theta, int N, int M, int Z, int num_total)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;

  int index = (N*M)*k + i + N*j;

   if (index < num_total)	{
       Input[index] += theta*(Input[index] - Input_old[index]);
   }
}

__global__ void PDResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int index = xIndex + N*yIndex;

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

__global__ void PDResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
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
////////////MAIN HOST FUNCTION ///////////////
extern "C" int TV_PD_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil, float lipschitz_const, int methodTV, int nonneg, int gpu_device, int dimX, int dimY, int dimZ)
{
   int deviceCount = -1; // number of devices
   cudaGetDeviceCount(&deviceCount);
   if (deviceCount == 0) {
       fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }

   int count = 0, i;
   float re, sigma, theta, lt, tau;
   re = 0.0f;

   tau = lambdaPar*0.1f;
   sigma = 1.0/(lipschitz_const*tau);
   theta = 1.0f;
   lt = tau/lambdaPar;

   if (dimZ <= 1) {
   /*2D verson*/
     int ImSize = dimX*dimY;
     float *d_input, *d_update, *d_old=NULL, *P1=NULL, *P2=NULL;

     dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
     dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));

      /*allocate space for images on device*/
      checkCudaErrors( cudaMalloc((void**)&d_input,ImSize*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&d_update,ImSize*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&d_old,ImSize*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&P1,ImSize*sizeof(float)) );
      checkCudaErrors( cudaMalloc((void**)&P2,ImSize*sizeof(float)) );

       checkCudaErrors( cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
       checkCudaErrors( cudaMemcpy(d_update,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
       cudaMemset(P1, 0, ImSize*sizeof(float));
       cudaMemset(P2, 0, ImSize*sizeof(float));

       /********************** Run CUDA 2D kernel here ********************/
       /* The main kernel */
       for (i = 0; i < iter; i++) {

           /* computing the the dual P variable */
           dualPD_kernel<<<dimGrid,dimBlock>>>(d_update, P1, P2, sigma, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           if (nonneg != 0) {
           PDnonneg2D_kernel<<<dimGrid,dimBlock>>>(d_update, dimX, dimY, ImSize);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() ); }

           /* projection step */
           if (methodTV == 0) Proj_funcPD2D_iso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, ImSize); /*isotropic TV*/
           else Proj_funcPD2D_aniso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, ImSize); /*anisotropic TV*/
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           /* copy U to U_old */
           PDcopy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_old, dimX, dimY, ImSize);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           /* calculate divergence */
           DivProj2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, P1, P2, lt, tau, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           if ((epsil != 0.0f) && (i % 5 == 0)) {
               /* calculate norm - stopping rules using the Thrust library */
               PDResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_old, P1, dimX, dimY, ImSize);
               checkCudaErrors( cudaDeviceSynchronize() );
               checkCudaErrors(cudaPeekAtLastError() );

               // setup arguments
               square<float>        unary_op;
               thrust::plus<float> binary_op;
               thrust::device_vector<float> d_vec(P1, P1 + ImSize);
               float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
               thrust::device_vector<float> d_vec2(d_update, d_update + ImSize);
               float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

               // compute norm
               re = (reduction/reduction2);
               if (re < epsil)  count++;
               if (count > 3) break;
             }

           getU2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_old, theta, dimX, dimY, ImSize);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );
          }
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input);
           cudaFree(d_update);
           cudaFree(d_old);
           cudaFree(P1);
           cudaFree(P2);

   }
   else {
           /*3D verson*/
           int ImSize = dimX*dimY*dimZ;

           /* adapted to work with up to 4 GPU devices in parallel */
           float *d_input0, *d_update0, *d_old0=NULL, *P1_0=NULL, *P2_0=NULL, *P3_0=NULL;
           float *d_input1, *d_update1, *d_old1=NULL, *P1_1=NULL, *P2_1=NULL, *P3_1=NULL;
           float *d_input2, *d_update2, *d_old2=NULL, *P1_2=NULL, *P2_2=NULL, *P3_2=NULL;
           float *d_input3, *d_update3, *d_old3=NULL, *P1_3=NULL, *P2_3=NULL, *P3_3=NULL;

           dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
           dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKZSIZE));

           cudaSetDevice(gpu_device);
           if (gpu_device == 0) {
           /*allocate space for images on device*/
           checkCudaErrors( cudaMalloc((void**)&d_input0,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_update0,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_old0,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P1_0,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P2_0,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P3_0,ImSize*sizeof(float)) );

            checkCudaErrors( cudaMemcpy(d_input0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            checkCudaErrors( cudaMemcpy(d_update0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            cudaMemset(P1_0, 0, ImSize*sizeof(float));
            cudaMemset(P2_0, 0, ImSize*sizeof(float));
            cudaMemset(P3_0, 0, ImSize*sizeof(float));
            }

           if (gpu_device == 1) {
           /*allocate space for images on device*/
           checkCudaErrors( cudaMalloc((void**)&d_input1,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_update1,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_old1,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P1_1,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P2_1,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P3_1,ImSize*sizeof(float)) );

            checkCudaErrors( cudaMemcpy(d_input1,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            checkCudaErrors( cudaMemcpy(d_update1,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            cudaMemset(P1_1, 0, ImSize*sizeof(float));
            cudaMemset(P2_1, 0, ImSize*sizeof(float));
            cudaMemset(P3_1, 0, ImSize*sizeof(float));
            }

           if (gpu_device == 2) {
           /*allocate space for images on device*/
           checkCudaErrors( cudaMalloc((void**)&d_input2,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_update2,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_old2,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P1_2,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P2_2,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P3_2,ImSize*sizeof(float)) );

            checkCudaErrors( cudaMemcpy(d_input2,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            checkCudaErrors( cudaMemcpy(d_update2,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            cudaMemset(P1_2, 0, ImSize*sizeof(float));
            cudaMemset(P2_2, 0, ImSize*sizeof(float));
            cudaMemset(P3_2, 0, ImSize*sizeof(float));
            }
           if (gpu_device == 3) {
           /*allocate space for images on device*/
           checkCudaErrors( cudaMalloc((void**)&d_input3,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_update3,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&d_old3,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P1_3,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P2_3,ImSize*sizeof(float)) );
           checkCudaErrors( cudaMalloc((void**)&P3_3,ImSize*sizeof(float)) );

            checkCudaErrors( cudaMemcpy(d_input3,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            checkCudaErrors( cudaMemcpy(d_update3,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
            cudaMemset(P1_3, 0, ImSize*sizeof(float));
            cudaMemset(P2_3, 0, ImSize*sizeof(float));
            cudaMemset(P3_3, 0, ImSize*sizeof(float));
            }


           /********************** Run CUDA 3D kernel here ********************/
       for (i = 0; i < iter; i++) {

         /* computing the the dual P variable */
          if (gpu_device == 0) dualPD3D_kernel<<<dimGrid,dimBlock>>>(d_update0, P1_0, P2_0, P3_0, sigma, dimX, dimY, dimZ);
          if (gpu_device == 1) dualPD3D_kernel<<<dimGrid,dimBlock>>>(d_update1, P1_1, P2_1, P3_1, sigma, dimX, dimY, dimZ);
          if (gpu_device == 2) dualPD3D_kernel<<<dimGrid,dimBlock>>>(d_update2, P1_2, P2_2, P3_2, sigma, dimX, dimY, dimZ);
          if (gpu_device == 3) dualPD3D_kernel<<<dimGrid,dimBlock>>>(d_update3, P1_3, P2_3, P3_3, sigma, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         if (nonneg != 0) {
        if (gpu_device == 0) PDnonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update0, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 1) PDnonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update1, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 2) PDnonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update2, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 3) PDnonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update3, dimX, dimY, dimZ, ImSize);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() ); }

         /* projection step */
         if (methodTV == 0) {
          if (gpu_device == 0) Proj_funcPD3D_iso_kernel<<<dimGrid,dimBlock>>>(P1_0, P2_0, P3_0, dimX, dimY, dimZ, ImSize); /*isotropic TV*/
          if (gpu_device == 1) Proj_funcPD3D_iso_kernel<<<dimGrid,dimBlock>>>(P1_1, P2_1, P3_1, dimX, dimY, dimZ, ImSize); /*isotropic TV*/
          if (gpu_device == 2) Proj_funcPD3D_iso_kernel<<<dimGrid,dimBlock>>>(P1_2, P2_2, P3_2, dimX, dimY, dimZ, ImSize); /*isotropic TV*/
          if (gpu_device == 3) Proj_funcPD3D_iso_kernel<<<dimGrid,dimBlock>>>(P1_3, P2_3, P3_3, dimX, dimY, dimZ, ImSize); /*isotropic TV*/
          }
         else {
          if (gpu_device == 0) Proj_funcPD3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1_0, P2_0, P3_0, dimX, dimY, dimZ, ImSize); /*anisotropic TV*/
          if (gpu_device == 1) Proj_funcPD3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1_1, P2_1, P3_1, dimX, dimY, dimZ, ImSize); /*anisotropic TV*/
          if (gpu_device == 2) Proj_funcPD3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1_2, P2_2, P3_2, dimX, dimY, dimZ, ImSize); /*anisotropic TV*/
          if (gpu_device == 3) Proj_funcPD3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1_3, P2_3, P3_3, dimX, dimY, dimZ, ImSize); /*anisotropic TV*/
          }
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         /* copy U to U_old */
        if (gpu_device == 0) PDcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update0, d_old0, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 1) PDcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update1, d_old1, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 2) PDcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update2, d_old2, dimX, dimY, dimZ, ImSize);
        if (gpu_device == 3) PDcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update3, d_old3, dimX, dimY, dimZ, ImSize);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         /* calculate divergence */
        if (gpu_device == 0) DivProj3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_input0, P1_0, P2_0, P3_0, lt, tau, dimX, dimY, dimZ);
        if (gpu_device == 1) DivProj3D_kernel<<<dimGrid,dimBlock>>>(d_update1, d_input1, P1_1, P2_1, P3_1, lt, tau, dimX, dimY, dimZ);
        if (gpu_device == 2) DivProj3D_kernel<<<dimGrid,dimBlock>>>(d_update2, d_input2, P1_2, P2_2, P3_2, lt, tau, dimX, dimY, dimZ);
        if (gpu_device == 3) DivProj3D_kernel<<<dimGrid,dimBlock>>>(d_update3, d_input3, P1_3, P2_3, P3_3, lt, tau, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         if (gpu_device == 0) {
            if ((epsil != 0.0f) && (i % 5 == 0)) {
           /* calculate norm - stopping rules using the Thrust library */
           PDResidCalc3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_old0, P1_0, dimX, dimY, dimZ, ImSize);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

          // setup arguments
           square<float>        unary_op;
           thrust::plus<float> binary_op;
           thrust::device_vector<float> d_vec(P1_0, P1_0 + ImSize);
           float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
           thrust::device_vector<float> d_vec2(d_update0, d_update0 + ImSize);
           float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

             // compute norm
             re = (reduction/reduction2);
             if (re < epsil)  count++;
             if (count > 3) break;
             }
          }

           /* get U*/
          if (gpu_device == 0) getU3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_old0, theta, dimX, dimY, dimZ, ImSize);
          if (gpu_device == 1) getU3D_kernel<<<dimGrid,dimBlock>>>(d_update1, d_old1, theta, dimX, dimY, dimZ, ImSize);
          if (gpu_device == 2) getU3D_kernel<<<dimGrid,dimBlock>>>(d_update2, d_old2, theta, dimX, dimY, dimZ, ImSize);
          if (gpu_device == 3) getU3D_kernel<<<dimGrid,dimBlock>>>(d_update3, d_old3, theta, dimX, dimY, dimZ, ImSize);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );
         }
           /***************************************************************/
           if (gpu_device == 0) {
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update0,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input0);
           cudaFree(d_update0);
           cudaFree(d_old0);
           cudaFree(P1_0);
           cudaFree(P2_0);
           cudaFree(P3_0);
          }
           if (gpu_device == 1) {
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update1,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input1);
           cudaFree(d_update1);
           cudaFree(d_old1);
           cudaFree(P1_1);
           cudaFree(P2_1);
           cudaFree(P3_1);
          }
           if (gpu_device == 2) {
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update2,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input2);
           cudaFree(d_update2);
           cudaFree(d_old2);
           cudaFree(P1_2);
           cudaFree(P2_2);
           cudaFree(P3_2);
          }
           if (gpu_device == 3) {
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update3,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input3);
           cudaFree(d_update3);
           cudaFree(d_old3);
           cudaFree(P1_3);
           cudaFree(P2_3);
           cudaFree(P3_3);
          }
   }
   //cudaDeviceReset();
   /*adding info into info_vector */
   infovector[0] = (float)(i);  /*iterations number (if stopped earlier based on tolerance)*/
   infovector[1] = re;  /* reached tolerance */
   cudaDeviceSynchronize();
   return 0;
}

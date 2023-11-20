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

#include "headers/TV_PD_GPU_core.h"
#include "cuda_kernels/TV_PD_GPU_kernels.cuh"
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
 * 8. GPU device number if for multigpu run (default 0)

 * Output:
 * [1] TV - Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * [1] Antonin Chambolle, Thomas Pock. "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", 2010
 */

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
/////////////////////////////////////////////////
///////////////// HOST FUNCTION /////////////////
/////////////////////////////////////////////////
extern "C" int TV_PD_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil, float lipschitz_const, int methodTV, int nonneg, int gpu_device, int dimX, int dimY, int dimZ)
{
   int deviceCount = -1; // number of devices
   cudaGetDeviceCount(&deviceCount);
   if (deviceCount == 0) {
       fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }
   checkCudaErrors(cudaSetDevice(gpu_device));

   int count = 0, i;
   float re, sigma, theta, lt, tau;
   re = 0.0f;

   tau = lambdaPar*0.1f;
   sigma = 1.0/(lipschitz_const*tau);
   theta = 1.0f;
   lt = tau/lambdaPar;

   if (dimZ <= 1) 
   {
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
       for (i = 0; i < iter; i++) 
       {

           /* computing the the dual P variable */
           dualPD_kernel<<<dimGrid,dimBlock>>>(d_update, P1, P2, sigma, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           if (nonneg != 0) {
           PDnonneg2D_kernel<<<dimGrid,dimBlock>>>(d_update, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() ); }

           /* projection step */
           if (methodTV == 0) Proj_funcPD2D_iso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY); /*isotropic TV*/
           else Proj_funcPD2D_aniso_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY); /*anisotropic TV*/
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           /* copy U to U_old */
           PDcopy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_old, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           /* calculate divergence */
           DivProj2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, P1, P2, lt, tau, dimX, dimY);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );

           if ((epsil != 0.0f) && (i % 5 == 0)) 
           {
               /* calculate norm - stopping rules using the Thrust library */
               PDResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_old, P1, dimX, dimY);
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
           getU2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_old, theta, dimX, dimY);
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

           dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
           dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKZSIZE));

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

           /********************** Run CUDA 3D kernel here ********************/
       for (i = 0; i < iter; i++) {

         /* computing the the dual P variable */
         dualPD3D_kernel<<<dimGrid,dimBlock>>>(d_update0, P1_0, P2_0, P3_0, sigma, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

        if (nonneg != 0) {
        PDnonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update0, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() ); }

         /* projection step */
         if (methodTV == 0) Proj_funcPD3D_iso_kernel<<<dimGrid,dimBlock>>>(P1_0, P2_0, P3_0, dimX, dimY, dimZ); /*isotropic TV*/
         else Proj_funcPD3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1_0, P2_0, P3_0, dimX, dimY, dimZ); /*anisotropic TV*/
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         /* copy U to U_old */
        PDcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update0, d_old0, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );

         /* calculate divergence */
         DivProj3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_input0, P1_0, P2_0, P3_0, lt, tau, dimX, dimY, dimZ);
         checkCudaErrors( cudaDeviceSynchronize() );
         checkCudaErrors(cudaPeekAtLastError() );


          if ((epsil != 0.0f) && (i % 5 == 0)) {
           /* calculate norm - stopping rules using the Thrust library */
           PDResidCalc3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_old0, P1_0, dimX, dimY, dimZ);
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

           /* get U*/
          getU3D_kernel<<<dimGrid,dimBlock>>>(d_update0, d_old0, theta, dimX, dimY, dimZ);
           checkCudaErrors( cudaDeviceSynchronize() );
           checkCudaErrors(cudaPeekAtLastError() );
         }
           /***************************************************************/
           //copy result matrix from device to host memory
           cudaMemcpy(Output,d_update0,ImSize*sizeof(float),cudaMemcpyDeviceToHost);

           cudaFree(d_input0);
           cudaFree(d_update0);
           cudaFree(d_old0);
           cudaFree(P1_0);
           cudaFree(P2_0);
           cudaFree(P3_0);

   }
   /*adding info into info_vector */
   infovector[0] = (float)(i);  /*iterations number (if stopped earlier based on tolerance)*/
   infovector[1] = re;  /* reached tolerance */
   checkCudaErrors( cudaDeviceSynchronize() );
   return 0;
}

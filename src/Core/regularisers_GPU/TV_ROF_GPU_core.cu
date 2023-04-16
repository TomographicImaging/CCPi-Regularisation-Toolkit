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

#include "headers/TV_ROF_GPU_core.h"
#include "cuda_kernels/TV_ROF_GPU_kernels.cuh"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

/* CUDA implementation of ROF-TV denoising/regularisation model [1] (2D/3D case)
*
* Input Parameters:
* 1. Noisy image/volume [REQUIRED]
* 2. lambda - regularisation parameter (scalar)
* 3. tau - marching step for explicit scheme, ~1 is recommended [REQUIRED]
* 4. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
* 5. eplsilon: tolerance constant
* 6. GPU device number if for multigpu run (default 0)

* Output:
* [1] Regularised image/volume
* [2] Information vector which contains [iteration no., reached tolerance]
*
* This function is based on the paper by
* [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
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
extern "C" int TV_ROF_GPU_main(float* Input, float* Output, float *infovector, float lambdaPar, int iter, float tau, float epsil, int gpu_device, int N, int M, int Z)
{
     int deviceCount = -1; // number of devices
     cudaGetDeviceCount(&deviceCount);
     if (deviceCount == 0) {
         fprintf(stderr, "No CUDA devices found\n");
          return -1;
      }
    checkCudaErrors(cudaSetDevice(gpu_device));

    float re;
    re = 0.0f;
	  int ImSize, count, n;
	  count = 0; n = 0;
    float *d_input, *d_update, *d_D1, *d_D2, *d_update_prev=NULL;

	if (Z == 0) Z = 1;
	      ImSize = N*M*Z;
        CHECK(cudaMalloc((void**)&d_input,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_update,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_D1,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_D2,ImSize*sizeof(float)));
        if (epsil != 0.0f) checkCudaErrors( cudaMalloc((void**)&d_update_prev,ImSize*sizeof(float)) );

        CHECK(cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_update,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));

        if (Z == 1) {
             // TV - 2D case
            dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
            dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));

            for(n=0; n < iter; n++) {

              if ((epsil != 0.0f) && (n % 5 == 0)) {
              ROFcopy_kernel2D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, N, M, ImSize);
              checkCudaErrors( cudaDeviceSynchronize() );
              checkCudaErrors(cudaPeekAtLastError() );
              }
                /* calculate differences */
                D1_func2D<<<dimGrid,dimBlock>>>(d_update, d_D1, N, M);
                CHECK(cudaDeviceSynchronize());
		            D2_func2D<<<dimGrid,dimBlock>>>(d_update, d_D2, N, M);
                CHECK(cudaDeviceSynchronize());
                /*running main kernel*/
                TV_kernel2D<<<dimGrid,dimBlock>>>(d_D1, d_D2, d_update, d_input, lambdaPar, tau, N, M);
                CHECK(cudaDeviceSynchronize());

                if ((epsil != 0.0f) && (n % 5 == 0)) {
                /* calculate norm - stopping rules using the Thrust library */
                ROFResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_update, d_update_prev, d_D1, N, M, ImSize);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors( cudaPeekAtLastError() );

                // setup arguments
		            square<float>        unary_op;
		            thrust::plus<float> binary_op;
                thrust::device_vector<float> d_vec(d_D1, d_D1 + ImSize);
		            float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
                thrust::device_vector<float> d_vec2(d_update, d_update + ImSize);
      		      float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

                // compute norm
                re = (reduction/reduction2);
                if (re < epsil)  count++;
                if (count > 3) break;
           	}

            }
        }
	 else {
	           // TV - 3D case
            dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
            dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));

            float *d_D3;
            CHECK(cudaMalloc((void**)&d_D3,N*M*Z*sizeof(float)));

            for(n=0; n < iter; n++) {

              if ((epsil != 0.0f) && (n % 5 == 0)) {
              ROFcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, N, M, Z, ImSize);
              checkCudaErrors( cudaDeviceSynchronize() );
              checkCudaErrors(cudaPeekAtLastError() );
              }
                /* calculate differences */
                D1_func3D<<<dimGrid,dimBlock>>>(d_update, d_D1, N, M, Z);
                CHECK(cudaDeviceSynchronize());
		            D2_func3D<<<dimGrid,dimBlock>>>(d_update, d_D2, N, M, Z);
                CHECK(cudaDeviceSynchronize());
                D3_func3D<<<dimGrid,dimBlock>>>(d_update, d_D3, N, M, Z);
                CHECK(cudaDeviceSynchronize());
                /*running main kernel*/
                TV_kernel3D<<<dimGrid,dimBlock>>>(d_D1, d_D2, d_D3, d_update, d_input, lambdaPar, tau, N, M, Z);
                CHECK(cudaDeviceSynchronize());

                if ((epsil != 0.0f) && (n % 5 == 0)) {
                /* calculate norm - stopping rules using the Thrust library */
                ROFResidCalc3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_update_prev, d_D1, N, M, Z, ImSize);
                checkCudaErrors( cudaDeviceSynchronize() );
                checkCudaErrors( cudaPeekAtLastError() );

                // setup arguments
                square<float>        unary_op;
                thrust::plus<float> binary_op;
                thrust::device_vector<float> d_vec(d_D1, d_D1 + ImSize);
                float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
                thrust::device_vector<float> d_vec2(d_update, d_update + ImSize);
                float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

                // compute norm
                re = (reduction/reduction2);
                if (re < epsil)  count++;
                if (count > 3) break;
              }
            }
            CHECK(cudaFree(d_D3));
        }
        CHECK(cudaMemcpy(Output,d_update,N*M*Z*sizeof(float),cudaMemcpyDeviceToHost));
        if (epsil != 0.0f) cudaFree(d_update_prev);
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_update));
        CHECK(cudaFree(d_D1));
        CHECK(cudaFree(d_D2));

	      infovector[0] = (float)(n);  /*iterations number (if stopped earlier based on tolerance)*/
        infovector[1] = re;  /* reached tolerance */
        checkCudaErrors( cudaDeviceSynchronize() );
        return 0;
}

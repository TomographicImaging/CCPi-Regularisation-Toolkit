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

#include "Diffus_4thO_GPU_core.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

/* CUDA implementation of fourth-order diffusion scheme [1] for piecewise-smooth recovery (2D/3D case)
 * The minimisation is performed using explicit scheme.
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Edge-preserving parameter (sigma)
 * 4. Number of iterations, for explicit scheme >= 150 is recommended
 * 5. tau - time-marching step for explicit scheme
 * 6. eplsilon: tolerance constant
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Hajiaboli, M.R., 2011. An anisotropic fourth-order diffusion filter for image noise removal. International Journal of Computer Vision, 92(2), pp.177-191.
 */

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16
#define EPS 1.0e-7
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void Weighted_Laplc2D_kernel(float *W_Lapl, float *U0, float sigma, int dimX, int dimY)
{
		int i1,i2,j1,j2;
		float gradX, gradX_sq, gradY, gradY_sq, gradXX, gradYY, gradXY, xy_2, denom, V_norm, V_orth, c, c_sq;

				int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        int index = i + dimX*j;

        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {

            /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == dimX) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;

				gradX = 0.5f*(U0[j*dimX+i2] - U0[j*dimX+i1]);
				gradX_sq = powf(gradX,2);

				gradY = 0.5f*(U0[j2*dimX+i] - U0[j1*dimX+i]);
                gradY_sq = powf(gradY,2);

                gradXX = U0[j*dimX+i2] + U0[j*dimX+i1] - 2*U0[index];
                gradYY = U0[j2*dimX+i] + U0[j1*dimX+i] - 2*U0[index];

                gradXY = 0.25f*(U0[j2*dimX+i2] + U0[j1*dimX+i1] - U0[j1*dimX+i2] - U0[j2*dimX+i1]);
                xy_2 = 2.0f*gradX*gradY*gradXY;

                denom =  gradX_sq + gradY_sq;

                if (denom <= EPS) {
                    V_norm = (gradXX*gradX_sq + xy_2 + gradYY*gradY_sq)/EPS;
                    V_orth = (gradXX*gradY_sq - xy_2 + gradYY*gradX_sq)/EPS;
                    }
                else  {
                    V_norm = (gradXX*gradX_sq + xy_2 + gradYY*gradY_sq)/denom;
                    V_orth = (gradXX*gradY_sq - xy_2 + gradYY*gradX_sq)/denom;
                    }

                c = 1.0f/(1.0f + denom/sigma);
                c_sq = c*c;

                W_Lapl[index] = c_sq*V_norm + c*V_orth;
		}
	return;
}

__global__ void Diffusion_update_step2D_kernel(float *Output, float *Input, float *W_Lapl, float lambdaPar, float sigmaPar2, float tau, int dimX, int dimY)
{
	int i1,i2,j1,j2;
    float gradXXc, gradYYc;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        int index = i + dimX*j;

        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {

            /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == dimX) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;

                    gradXXc = W_Lapl[j*dimX+i2] + W_Lapl[j*dimX+i1] - 2*W_Lapl[index];
                    gradYYc = W_Lapl[j2*dimX+i] + W_Lapl[j1*dimX+i] - 2*W_Lapl[index];

                    Output[index] += tau*(-lambdaPar*(gradXXc + gradYYc) - (Output[index] - Input[index]));
		}
	return;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
__global__ void Weighted_Laplc3D_kernel(float *W_Lapl, float *U0, float sigma, int dimX, int dimY, int dimZ)
{
		int i1,i2,j1,j2,k1,k2;
		float gradX, gradX_sq, gradY, gradY_sq, gradXX, gradYY, gradXY, xy_2, denom, V_norm, V_orth, c, c_sq, gradZ, gradZ_sq, gradZZ, gradXZ, gradYZ, xyz_1, xyz_2;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		int k = blockDim.z * blockIdx.z + threadIdx.z;

		if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {

		    /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == dimX) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
			k1 = k+1; if (k1 == dimZ) k1 = k-1;
			k2 = k-1; if (k2 < 0) k2 = k+1;

				int index = (dimX*dimY)*k + j*dimX+i;

				gradX = 0.5f*(U0[(dimX*dimY)*k + j*dimX+i2] - U0[(dimX*dimY)*k + j*dimX+i1]);
				gradX_sq = pow(gradX,2);

				gradY = 0.5f*(U0[(dimX*dimY)*k + j2*dimX+i] - U0[(dimX*dimY)*k + j1*dimX+i]);
                gradY_sq = pow(gradY,2);

                gradZ = 0.5f*(U0[(dimX*dimY)*k2 + j*dimX+i] - U0[(dimX*dimY)*k1 + j*dimX+i]);
                gradZ_sq = pow(gradZ,2);

                gradXX = U0[(dimX*dimY)*k + j*dimX+i2] + U0[(dimX*dimY)*k + j*dimX+i1] - 2*U0[index];
                gradYY = U0[(dimX*dimY)*k + j2*dimX+i] + U0[(dimX*dimY)*k + j1*dimX+i] - 2*U0[index];
                gradZZ = U0[(dimX*dimY)*k2 + j*dimX+i] + U0[(dimX*dimY)*k1 + j*dimX+i] - 2*U0[index];

                gradXY = 0.25f*(U0[(dimX*dimY)*k + j2*dimX+i2] + U0[(dimX*dimY)*k + j1*dimX+i1] - U0[(dimX*dimY)*k + j1*dimX+i2] - U0[(dimX*dimY)*k + j2*dimX+i1]);
                gradXZ = 0.25f*(U0[(dimX*dimY)*k2 + j*dimX+i2] - U0[(dimX*dimY)*k2+j*dimX+i1] - U0[(dimX*dimY)*k1+j*dimX+i2] + U0[(dimX*dimY)*k1+j*dimX+i1]);
                gradYZ = 0.25f*(U0[(dimX*dimY)*k2 +j2*dimX+i] - U0[(dimX*dimY)*k2+j1*dimX+i] - U0[(dimX*dimY)*k1+j2*dimX+i] + U0[(dimX*dimY)*k1+j1*dimX+i]);

                xy_2  = 2.0f*gradX*gradY*gradXY;
                xyz_1 = 2.0f*gradX*gradZ*gradXZ;
                xyz_2 = 2.0f*gradY*gradZ*gradYZ;

                denom =  gradX_sq + gradY_sq + gradZ_sq;

					if (denom <= EPS) {
					V_norm = (gradXX*gradX_sq + gradYY*gradY_sq + gradZZ*gradZ_sq + xy_2 + xyz_1 + xyz_2)/EPS;
                    V_orth = ((gradY_sq + gradZ_sq)*gradXX + (gradX_sq + gradZ_sq)*gradYY + (gradX_sq + gradY_sq)*gradZZ - xy_2 - xyz_1 - xyz_2)/EPS;
					}
					else  {
					V_norm = (gradXX*gradX_sq + gradYY*gradY_sq + gradZZ*gradZ_sq + xy_2 + xyz_1 + xyz_2)/denom;
                    V_orth = ((gradY_sq + gradZ_sq)*gradXX + (gradX_sq + gradZ_sq)*gradYY + (gradX_sq + gradY_sq)*gradZZ - xy_2 - xyz_1 - xyz_2)/denom;
					}

                c = 1.0f/(1.0f + denom/sigma);
                c_sq = c*c;

            W_Lapl[index] = c_sq*V_norm + c*V_orth;
		}
	return;
}
__global__ void Diffusion_update_step3D_kernel(float *Output, float *Input, float *W_Lapl, float lambdaPar, float sigmaPar2, float tau, int dimX, int dimY, int dimZ)
{
	int i1,i2,j1,j2,k1,k2;
    float gradXXc, gradYYc, gradZZc;

		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		int k = blockDim.z * blockIdx.z + threadIdx.z;

		if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {

		    /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == dimX) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
			k1 = k+1; if (k1 == dimZ) k1 = k-1;
			k2 = k-1; if (k2 < 0) k2 = k+1;

			int index = (dimX*dimY)*k + j*dimX+i;

                    gradXXc = W_Lapl[(dimX*dimY)*k + j*dimX+i2] + W_Lapl[(dimX*dimY)*k + j*dimX+i1] - 2*W_Lapl[index];
                    gradYYc = W_Lapl[(dimX*dimY)*k + j2*dimX+i] + W_Lapl[(dimX*dimY)*k + j1*dimX+i] - 2*W_Lapl[index];
                    gradZZc = W_Lapl[(dimX*dimY)*k2 + j*dimX+i] + W_Lapl[(dimX*dimY)*k1 + j*dimX+i] - 2*W_Lapl[index];

                    Output[index] += tau*(-lambdaPar*(gradXXc + gradYYc + gradZZc) - (Output[index] - Input[index]));
		}
	return;
}

__global__ void Diff4thcopy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int index = xIndex + N*yIndex;

    if (index < num_total)	{
        Output[index] = Input[index];
    }
}


__global__ void Diff4thResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int index = xIndex + N*yIndex;

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

__global__ void Diff4thcopy_kernel3D(float *Input, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (N*M)*k + i + N*j;

    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

__global__ void Diff4thResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
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
extern "C" int Diffus4th_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, float epsil, int N, int M, int Z)
{

    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
      }

        int dimTotal, n, count = 0;
        float *d_input, *d_output, *d_W_Lapl, *d_update_prev=NULL, re;
        re = 0.0f;
        float sigmaPar2;
        sigmaPar2 = sigmaPar*sigmaPar;
        dimTotal = N*M*Z;

        CHECK(cudaMalloc((void**)&d_input,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_W_Lapl,dimTotal*sizeof(float)));
        if (epsil != 0.0f) checkCudaErrors( cudaMalloc((void**)&d_update_prev,dimTotal*sizeof(float)) );

        CHECK(cudaMemcpy(d_input,Input,dimTotal*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output,Input,dimTotal*sizeof(float),cudaMemcpyHostToDevice));

       /*2D case */
       dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
       dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
       dim3 dimBlock3(BLKXSIZE,BLKYSIZE,BLKZSIZE);
       dim3 dimGrid3(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKZSIZE));


        for(n=0; n < iterationsNumb; n++) {

					if ((epsil != 0.0f) && (n % 5 == 0)) {
						if (Z == 1) Diff4thcopy_kernel2D<<<dimGrid,dimBlock>>>(d_output, d_update_prev, N, M, dimTotal);
						else Diff4thcopy_kernel3D<<<dimGrid3,dimBlock3>>>(d_output, d_update_prev, N, M, Z, dimTotal);
						checkCudaErrors( cudaDeviceSynchronize() );
						checkCudaErrors(cudaPeekAtLastError() );
					}

          if (Z == 1) {
        	     /*2D case */
				       /* Calculating weighted Laplacian */
				      Weighted_Laplc2D_kernel<<<dimGrid,dimBlock>>>(d_W_Lapl, d_output, sigmaPar2, N, M);
				      CHECK(cudaDeviceSynchronize());
				      /* Perform iteration step */
				      Diffusion_update_step2D_kernel<<<dimGrid,dimBlock>>>(d_output, d_input, d_W_Lapl, lambdaPar, sigmaPar2, tau, N, M);
				      CHECK(cudaDeviceSynchronize());
                }
            else {
              /* Calculating weighted Laplacian */
              Weighted_Laplc3D_kernel<<<dimGrid3,dimBlock3>>>(d_W_Lapl, d_output, sigmaPar2, N, M, Z);
              CHECK(cudaDeviceSynchronize());
              /* Perform iteration step */
              Diffusion_update_step3D_kernel<<<dimGrid3,dimBlock3>>>(d_output, d_input, d_W_Lapl, lambdaPar, sigmaPar2, tau, N, M, Z);
              CHECK(cudaDeviceSynchronize());
            }
            if ((epsil != 0.0f) && (n % 5 == 0)) {
            /* calculate norm - stopping rules using the Thrust library */
            if (Z == 1) Diff4thResidCalc2D_kernel<<<dimGrid,dimBlock>>>(d_output, d_update_prev, d_W_Lapl, N, M, dimTotal);
            else Diff4thResidCalc3D_kernel<<<dimGrid3,dimBlock3>>>(d_output, d_update_prev, d_W_Lapl, N, M, Z, dimTotal);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors( cudaPeekAtLastError() );

            // setup arguments
            square<float>        unary_op;
            thrust::plus<float> binary_op;
            thrust::device_vector<float> d_vec(d_W_Lapl, d_W_Lapl + dimTotal);
            float reduction = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0.0f, binary_op));
            thrust::device_vector<float> d_vec2(d_output, d_output + dimTotal);
            float reduction2 = std::sqrt(thrust::transform_reduce(d_vec2.begin(), d_vec2.end(), unary_op, 0.0f, binary_op));

            // compute norm
            re = (reduction/reduction2);
            if (re < epsil)  count++;
            if (count > 3) break;
            }
	       }

        CHECK(cudaMemcpy(Output,d_output,dimTotal*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        CHECK(cudaFree(d_W_Lapl));
        if (epsil != 0.0f) cudaFree(d_update_prev);

        /*adding info into info_vector */
        infovector[0] = (float)(n);  /*iterations number (if stopped earlier based on tolerance)*/
        infovector[1] = re;  /* reached tolerance */
        return 0;
}

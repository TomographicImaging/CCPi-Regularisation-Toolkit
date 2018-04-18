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

#include "NonlDiff_GPU_core.h"

/* CUDA implementation of linear and nonlinear diffusion with the regularisation model [1,2] (2D/3D case)
 * The minimisation is performed using explicit scheme. 
 *
 * Input Parameters:
 * 1. Noisy image/volume 
 * 2. lambda - regularization parameter
 * 3. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 4. Number of iterations, for explicit scheme >= 150 is recommended 
 * 5. tau - time-marching step for explicit scheme
 * 6. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
 *
 * Output:
 * [1] Regularized image/volume 
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 * [2] Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.
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
   
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void LinearDiff2D_kernel(float *Input, float *Output, float lambdaPar, float tau, int N, int M)
    {
		int i1,i2,j1,j2;
		float e,w,n,s,e1,w1,n1,s1;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + N*j;
        
        if ((i >= 0) && (i < N) && (j >= 0) && (j < M)) {
            
            /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == N) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == M) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            
		        e = Output[j*N+i1];
                w = Output[j*N+i2];
                n = Output[j1*N+i];
                s = Output[j2*N+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                
                Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1) - (Output[index] - Input[index])); 
		}
	} 
    
 __global__ void NonLinearDiff2D_kernel(float *Input, float *Output, float lambdaPar, float sigmaPar, float tau, int penaltytype, int N, int M)
    {
		int i1,i2,j1,j2;
		float e,w,n,s,e1,w1,n1,s1;
		int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + N*j;
        
        if ((i >= 0) && (i < N) && (j >= 0) && (j < M)) {
            
            /* boundary conditions (Neumann reflections) */
			i1 = i+1; if (i1 == N) i1 = i-1;
			i2 = i-1; if (i2 < 0) i2 = i+1;
            j1 = j+1; if (j1 == M) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            
		        e = Output[j*N+i1];
                w = Output[j*N+i2];
                n = Output[j1*N+i];
                s = Output[j2*N+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                
            if (penaltytype == 1){
            /* Huber penalty */
            if (abs(e1) > sigmaPar) e1 =  sign(e1);
            else e1 = e1/sigmaPar;
            
            if (abs(w1) > sigmaPar) w1 =  sign(w1);
            else w1 = w1/sigmaPar;
            
            if (abs(n1) > sigmaPar) n1 =  sign(n1);
            else n1 = n1/sigmaPar;
            
            if (abs(s1) > sigmaPar) s1 =  sign(s1);
            else s1 = s1/sigmaPar;
            }
            else if (penaltytype == 2) {
            /* Perona-Malik */
            e1 = (e1)/(1.0f + pow((e1/sigmaPar),2));
            w1 = (w1)/(1.0f + pow((w1/sigmaPar),2));
            n1 = (n1)/(1.0f + pow((n1/sigmaPar),2));
            s1 = (s1)/(1.0f + pow((s1/sigmaPar),2));
            }
            else if (penaltytype == 3) {
            /* Tukey Biweight */
            if (abs(e1) <= sigmaPar) e1 =  e1*pow((1.0f - pow((e1/sigmaPar),2)), 2);
            else e1 = 0.0f;
            if (abs(w1) <= sigmaPar) w1 =  w1*pow((1.0f - pow((w1/sigmaPar),2)), 2);
            else w1 = 0.0f;
            if (abs(n1) <= sigmaPar) n1 =  n1*pow((1.0f - pow((n1/sigmaPar),2)), 2);
            else n1 = 0.0f;
            if (abs(s1) <= sigmaPar) s1 =  s1*pow((1.0f - pow((s1/sigmaPar),2)), 2);
            else s1 = 0.0f;
            }
            else printf("%s \n", "No penalty function selected! Use 1,2 or 3.");
                            
            Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1) - (Output[index] - Input[index])); 
		}
	} 
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/

/////////////////////////////////////////////////
// HOST FUNCTION
extern "C" void NonlDiff_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int N, int M, int Z)
{
	    // set up device
		int dev = 0;
		CHECK(cudaSetDevice(dev));
        float *d_input, *d_output;
        float sigmaPar2;
        sigmaPar2 = sigmaPar/sqrt(2.0f);
        
	if (Z == 1) {
	     /*2D case */ 
        CHECK(cudaMalloc((void**)&d_input,N*M*Z*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output,N*M*Z*sizeof(float)));
                
        CHECK(cudaMemcpy(d_input,Input,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output,Input,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));      
        
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
             
        for(int n=0; n < iterationsNumb; n++) {
				if (sigmaPar == 0.0f) {
				/* linear diffusion (heat equation) */
				LinearDiff2D_kernel<<<dimGrid,dimBlock>>>(d_input, d_output, lambdaPar, tau, N, M);
				CHECK(cudaDeviceSynchronize());
				}
				else {
				/* nonlinear diffusion */
				NonLinearDiff2D_kernel<<<dimGrid,dimBlock>>>(d_input, d_output, lambdaPar, sigmaPar2, tau, penaltytype, N, M);
				CHECK(cudaDeviceSynchronize());
				}
        }
	}
	else {
		/*3D case*/
		}
        
        CHECK(cudaMemcpy(Output,d_output,N*M*Z*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        cudaDeviceReset(); 
}

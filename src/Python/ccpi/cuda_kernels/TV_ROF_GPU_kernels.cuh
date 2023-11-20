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

/* 
Raw CUDA Kernels for TV_ROF regularisation model
*/

#define EPS 1.0e-8

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__host__ __device__ int sign (float x)
{
        return (x > 0) - (x < 0);
}

/*********************2D case****************************/
    /* differences 1 */
extern "C" __global__ void D1_func2D(float* Input, float* D1, int N, int M)
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
          denom2 = 0.5f*(sign((float)NOMy_1) + sign((float)NOMy_0))*(MIN(fabs((float)NOMy_1), fabs((float)NOMy_0)));
          denom2 = denom2*denom2;
          T1 = sqrt(denom1 + denom2 + EPS);
          D1[index] = NOMx_1/T1;
		  }
	  }

    /* differences 2 */
extern "C" __global__ void D2_func2D(float* Input, float* D2, int N, int M)
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
                denom2 = 0.5f*(sign((float)NOMx_1) + sign((float)NOMx_0))*(MIN(fabs((float)NOMx_1), fabs((float)NOMx_0)));
                denom2 = denom2*denom2;
                T2 = sqrtf(denom1 + denom2 + EPS);
                D2[index] = NOMy_1/T2;
		}
	}

extern "C" __global__ void TV_kernel2D(float *D1, float *D2, float *Update, float *Input, float lambdaPar, float tau_step, int N, int M)
    {
    int i2, j2;
    float dv1, dv2;
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

        Update[index] += tau_step*(lambdaPar*(dv1 + dv2) - (Update[index] - Input[index]));        
        }
	}

extern "C" __global__ void ROFcopy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int index = xIndex + N*yIndex;

    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

extern "C" __global__ void ROFResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int index = xIndex + N*yIndex;

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}    
/*********************3D case****************************/
    /* differences 1 */
    extern "C" __global__ void D1_func3D(float* Input, float* D1, int dimX, int dimY, int dimZ)
    {
		float NOMx_1, NOMy_1, NOMy_0, NOMz_1, NOMz_0, denom1, denom2,denom3, T1;
		long long i1, i2, k1, j1, j2, k2;

        const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= dimX || j >= dimY || k >= dimZ)
            return;

        long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

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

    /* differences 2 */
    extern "C" __global__ void D2_func3D(float* Input, float* D2, int dimX, int dimY, int dimZ)
    {
		float NOMx_1, NOMy_1, NOMx_0, NOMz_1, NOMz_0, denom1, denom2, denom3, T2;
		long long i1,i2,k1,j1,j2,k2;

        const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= dimX || j >= dimY || k >= dimZ)
            return;

        long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

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

	  /* differences 3 */
    extern "C" __global__ void D3_func3D(float* Input, float* D3, int dimX, int dimY, int dimZ)
    {
		float NOMx_1, NOMy_1, NOMx_0, NOMy_0, NOMz_1, denom1, denom2, denom3, T3;
		long long i1,i2,k1,j1,j2,k2;

        const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= dimX || j >= dimY || k >= dimZ)
            return;

        long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

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

extern "C" __global__ void TV_kernel3D(float *D1, float *D2, float *D3, float *Update, float *Input, float lambdaPar, float tau, int dimX, int dimY, int dimZ)
    {	
        float dv1, dv2, dv3;
        long long i1,i2,k1,j1,j2,k2;

        const long i = blockDim.x * blockIdx.x + threadIdx.x;
        const long j = blockDim.y * blockIdx.y + threadIdx.y;
        const long k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= dimX || j >= dimY || k >= dimZ)
            return;

        long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

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

        Update[index] += tau*(lambdaPar*(dv1 + dv2 + dv3) - (Update[index] - Input[index]));		
	}

extern "C" __global__ void ROFcopy_kernel3D(float *Input, float* Output, int dimX, int dimY, int dimZ, int num_total)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

extern "C" __global__ void ROFResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int dimX, int dimY, int dimZ, int num_total)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}
/*****************************************************/
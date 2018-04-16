#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

/*Some CUDA functions which frequently re-used from various modules */
/***********************************************************************/
__global__ void copy_kernel2D(float *Input, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input[index];
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

__global__ void ResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

__global__ void ResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = (N*M)*k + i + N*j;
    
    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}


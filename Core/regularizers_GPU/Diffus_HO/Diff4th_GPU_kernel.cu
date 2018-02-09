#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "Diff4th_GPU_kernel.h"

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

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
#define sizeT (sizeX*sizeY*sizeZ)
#define epsilon 0.00000001

/////////////////////////////////////////////////
// 2D Image denosing - Second Step (The second derrivative)
__global__ void Diff4th2D_derriv(float* B, float* A, float *A0, int N, int M, float sigma, int iter, float tau, float lambda)
{
    float gradXXc = 0, gradYYc = 0;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    int index = j + i*N;
    
    if (((i < 1) || (i > N-2)) || ((j < 1) || (j > M-2))) {
        return;    }
    
    int indexN = (j)+(i-1)*(N); if (A[indexN] == 0) indexN = index;
    int indexS = (j)+(i+1)*(N); if (A[indexS] == 0) indexS = index;
    int indexW = (j-1)+(i)*(N); if (A[indexW] == 0) indexW = index;
    int indexE = (j+1)+(i)*(N); if (A[indexE] == 0) indexE = index;
    
    gradXXc = B[indexN] + B[indexS] - 2*B[index] ;
    gradYYc = B[indexW] + B[indexE] - 2*B[index] ;
    A[index]  = A[index] - tau*((A[index] - A0[index]) + lambda*(gradXXc + gradYYc));
}

// 2D Image denosing - The First Step
__global__ void Diff4th2D(float* A, float* B, int N, int M, float sigma, int iter, float tau)
{
    float gradX, gradX_sq, gradY, gradY_sq, gradXX, gradYY, gradXY, sq_sum, xy_2,  V_norm, V_orth, c, c_sq;
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    int index = j + i*N;
    
    V_norm = 0.0f; V_orth = 0.0f;
    
    if (((i < 1) || (i > N-2)) || ((j < 1) || (j > M-2))) {
        return;    }
    
    int indexN = (j)+(i-1)*(N); if (A[indexN] == 0) indexN = index;
    int indexS = (j)+(i+1)*(N); if (A[indexS] == 0) indexS = index;
    int indexW = (j-1)+(i)*(N); if (A[indexW] == 0) indexW = index;
    int indexE = (j+1)+(i)*(N); if (A[indexE] == 0) indexE = index;
    int indexNW = (j-1)+(i-1)*(N); if (A[indexNW] == 0) indexNW = index;
    int indexNE = (j+1)+(i-1)*(N); if (A[indexNE] == 0) indexNE = index;
    int indexWS = (j-1)+(i+1)*(N); if (A[indexWS] == 0) indexWS = index;
    int indexES = (j+1)+(i+1)*(N); if (A[indexES] == 0) indexES = index;
    
    gradX = 0.5f*(A[indexN]-A[indexS]);
    gradX_sq = gradX*gradX;
    gradXX = A[indexN] + A[indexS] - 2*A[index];
    
    gradY = 0.5f*(A[indexW]-A[indexE]);
    gradY_sq = gradY*gradY;
    gradYY = A[indexW] + A[indexE] - 2*A[index];
    
    gradXY = 0.25f*(A[indexNW] - A[indexNE] - A[indexWS] + A[indexES]);
    xy_2 = 2.0f*gradX*gradY*gradXY;
    sq_sum =  gradX_sq + gradY_sq;
    
    if (sq_sum <= epsilon) {
        V_norm = (gradXX*gradX_sq + xy_2 + gradYY*gradY_sq)/epsilon;
        V_orth = (gradXX*gradY_sq - xy_2 + gradYY*gradX_sq)/epsilon; }
    else  {
        V_norm = (gradXX*gradX_sq + xy_2 + gradYY*gradY_sq)/sq_sum;
        V_orth = (gradXX*gradY_sq - xy_2 + gradYY*gradX_sq)/sq_sum;  }
    
    c = 1.0f/(1.0f + sq_sum/sigma);
    c_sq = c*c;
    B[index] =  c_sq*V_norm + c*V_orth;
}

/////////////////////////////////////////////////
// 3D data parocerssing
__global__ void Diff4th3D_derriv(float *B, float *A, float *A0, int N, int M, int Z, float sigma, int iter, float tau, float lambda)
{
    float gradXXc = 0, gradYYc = 0, gradZZc = 0;
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = xIndex + M*yIndex + N*M*zIndex;
    
    if (((xIndex < 1) || (xIndex > N-2)) || ((yIndex < 1) || (yIndex > M-2)) || ((zIndex < 1) || (zIndex > Z-2))) {
        return;    }
    
    int indexN = (xIndex-1) + M*yIndex + N*M*zIndex; if (A[indexN] == 0) indexN = index;
    int indexS = (xIndex+1) + M*yIndex + N*M*zIndex; if (A[indexS] == 0) indexS = index;
    int indexW = xIndex + M*(yIndex-1) + N*M*zIndex; if (A[indexW] == 0) indexW = index;
    int indexE = xIndex + M*(yIndex+1) + N*M*zIndex; if (A[indexE] == 0) indexE = index;
    int indexU = xIndex + M*yIndex + N*M*(zIndex-1); if (A[indexU] == 0) indexU = index;
    int indexD = xIndex + M*yIndex + N*M*(zIndex+1); if (A[indexD] == 0) indexD = index;
    
    gradXXc = B[indexN] + B[indexS] - 2*B[index] ;
    gradYYc = B[indexW] + B[indexE] - 2*B[index] ;
    gradZZc = B[indexU] + B[indexD] - 2*B[index] ;   
        
    A[index]  = A[index] - tau*((A[index] - A0[index]) + lambda*(gradXXc + gradYYc + gradZZc));    
}

__global__ void Diff4th3D(float* A, float* B, int N, int M, int Z, float sigma, int iter, float tau)
{
    float gradX, gradX_sq, gradY, gradY_sq, gradZ, gradZ_sq, gradXX, gradYY, gradZZ, gradXY, gradXZ, gradYZ, sq_sum, xy_2, xyz_1, xyz_2, V_norm, V_orth, c, c_sq;
    
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
    
    int index = xIndex + M*yIndex + N*M*zIndex;
    V_norm = 0.0f; V_orth = 0.0f;
    
    if (((xIndex < 1) || (xIndex > N-2)) || ((yIndex < 1) || (yIndex > M-2)) || ((zIndex < 1) || (zIndex > Z-2))) {
        return;    }
    
    B[index] = 0;
    
    int indexN = (xIndex-1) + M*yIndex + N*M*zIndex; if (A[indexN] == 0) indexN = index;
    int indexS = (xIndex+1) + M*yIndex + N*M*zIndex; if (A[indexS] == 0) indexS = index;
    int indexW = xIndex + M*(yIndex-1) + N*M*zIndex; if (A[indexW] == 0) indexW = index;
    int indexE = xIndex + M*(yIndex+1) + N*M*zIndex; if (A[indexE] == 0) indexE = index;
    int indexU = xIndex + M*yIndex + N*M*(zIndex-1); if (A[indexU] == 0) indexU = index;
    int indexD = xIndex + M*yIndex + N*M*(zIndex+1); if (A[indexD] == 0) indexD = index;
    
    int indexNW = (xIndex-1) + M*(yIndex-1) + N*M*zIndex;  if (A[indexNW] == 0) indexNW = index;
    int indexNE = (xIndex-1) + M*(yIndex+1) + N*M*zIndex;  if (A[indexNE] == 0) indexNE = index;
    int indexWS =  (xIndex+1) + M*(yIndex-1) + N*M*zIndex; if (A[indexWS] == 0) indexWS = index;
    int indexES = (xIndex+1) + M*(yIndex+1) + N*M*zIndex;  if (A[indexES] == 0) indexES = index;
    
    int indexUW = (xIndex-1) + M*(yIndex) + N*M*(zIndex-1); if (A[indexUW] == 0) indexUW = index;
    int indexUE = (xIndex+1) + M*(yIndex) + N*M*(zIndex-1); if (A[indexUE] == 0) indexUE = index;
    int indexDW =  (xIndex-1) + M*(yIndex) + N*M*(zIndex+1); if (A[indexDW] == 0) indexDW = index;
    int indexDE = (xIndex+1) + M*(yIndex) + N*M*(zIndex+1); if (A[indexDE] == 0) indexDE = index;
    
    int indexUN = (xIndex) + M*(yIndex-1) + N*M*(zIndex-1);  if (A[indexUN] == 0) indexUN = index;
    int indexUS = (xIndex) + M*(yIndex+1) + N*M*(zIndex-1);  if (A[indexUS] == 0) indexUS = index;
    int indexDN =  (xIndex) + M*(yIndex-1) + N*M*(zIndex+1); if (A[indexDN] == 0) indexDN = index;
    int indexDS = (xIndex) + M*(yIndex+1) + N*M*(zIndex+1);  if (A[indexDS] == 0) indexDS = index;
    
    gradX = 0.5f*(A[indexN]-A[indexS]);
    gradX_sq = gradX*gradX;
    gradXX = A[indexN] + A[indexS] - 2*A[index];
    
    gradY = 0.5f*(A[indexW]-A[indexE]);
    gradY_sq = gradY*gradY;
    gradYY = A[indexW] + A[indexE] - 2*A[index];
    
    gradZ = 0.5f*(A[indexU]-A[indexD]);
    gradZ_sq = gradZ*gradZ;
    gradZZ = A[indexU] + A[indexD] - 2*A[index];
    
    gradXY = 0.25f*(A[indexNW] - A[indexNE] - A[indexWS] + A[indexES]);
    gradXZ = 0.25f*(A[indexUW] - A[indexUE] - A[indexDW] + A[indexDE]);
    gradYZ = 0.25f*(A[indexUN] - A[indexUS] - A[indexDN] + A[indexDS]);
    
    xy_2  = 2.0f*gradX*gradY*gradXY;
    xyz_1 = 2.0f*gradX*gradZ*gradXZ;
    xyz_2 = 2.0f*gradY*gradZ*gradYZ;
    
    sq_sum =  gradX_sq + gradY_sq + gradZ_sq;
    
    if (sq_sum <= epsilon) {
        V_norm = (gradXX*gradX_sq + gradYY*gradY_sq + gradZZ*gradZ_sq + xy_2 + xyz_1 + xyz_2)/epsilon;
        V_orth = ((gradY_sq + gradZ_sq)*gradXX + (gradX_sq + gradZ_sq)*gradYY + (gradX_sq + gradY_sq)*gradZZ - xy_2 - xyz_1 - xyz_2)/epsilon;  }
    else  {
        V_norm = (gradXX*gradX_sq + gradYY*gradY_sq + gradZZ*gradZ_sq + xy_2 + xyz_1 + xyz_2)/sq_sum;
        V_orth = ((gradY_sq + gradZ_sq)*gradXX + (gradX_sq + gradZ_sq)*gradYY + (gradX_sq + gradY_sq)*gradZZ - xy_2 - xyz_1 - xyz_2)/sq_sum;  }
    
    c = 1;
    if ((1.0f + sq_sum/sigma) != 0.0f)  {c = 1.0f/(1.0f + sq_sum/sigma);}
    
    c_sq = c*c;
    B[index] =  c_sq*V_norm + c*V_orth;
}

/******************************************************/
/********* HOST FUNCTION*************/
extern "C" void Diff4th_GPU_kernel(float* A, float* B, int N, int M, int Z, float sigma, int iter, float tau, float lambda)
{
     int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }    
    
      int BLKXSIZE, BLKYSIZE,BLKZSIZE;
      float *Ad, *Bd, *Cd;
      sigma = sigma*sigma;
    
    if (Z == 0){
        // 4th order diffusion for 2D case     
        BLKXSIZE = 8;
        BLKYSIZE = 16;
        
        dim3 dimBlock(BLKXSIZE,BLKYSIZE);
        dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE));
        
        checkCudaErrors(cudaMalloc((void**)&Ad,N*M*sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&Bd,N*M*sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&Cd,N*M*sizeof(float)));
        
        checkCudaErrors(cudaMemcpy(Ad,A,N*M*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(Bd,A,N*M*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(Cd,A,N*M*sizeof(float),cudaMemcpyHostToDevice));
        
        int n = 1;
        while (n <= iter) {
            Diff4th2D<<<dimGrid,dimBlock>>>(Bd, Cd, N, M, sigma, iter, tau);
            cudaDeviceSynchronize();
            checkCudaErrors( cudaPeekAtLastError() );
            Diff4th2D_derriv<<<dimGrid,dimBlock>>>(Cd, Bd, Ad, N, M, sigma, iter, tau, lambda);
            cudaDeviceSynchronize();
            checkCudaErrors( cudaPeekAtLastError() );
            n++;
        }
        checkCudaErrors(cudaMemcpy(B,Bd,N*M*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
    }
    
    if (Z != 0){
        // 4th order diffusion for 3D case
        BLKXSIZE = 8;
        BLKYSIZE = 8;
        BLKZSIZE = 8;        
        
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));
        
        checkCudaErrors(cudaMalloc((void**)&Ad,N*M*Z*sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&Bd,N*M*Z*sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&Cd,N*M*Z*sizeof(float)));
        
        checkCudaErrors(cudaMemcpy(Ad,A,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(Bd,A,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));        
        checkCudaErrors(cudaMemcpy(Cd,A,N*M*Z*sizeof(float),cudaMemcpyHostToDevice));
        
        int n = 1;
        while (n <= iter) {
            Diff4th3D<<<dimGrid,dimBlock>>>(Bd, Cd, N, M, Z, sigma, iter, tau);
            cudaDeviceSynchronize();
            checkCudaErrors( cudaPeekAtLastError() );
            Diff4th3D_derriv<<<dimGrid,dimBlock>>>(Cd, Bd, Ad, N, M, Z, sigma, iter, tau, lambda);
            cudaDeviceSynchronize();
            checkCudaErrors( cudaPeekAtLastError() );
            n++;
        }
        checkCudaErrors(cudaMemcpy(B,Bd,N*M*Z*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
    }
}

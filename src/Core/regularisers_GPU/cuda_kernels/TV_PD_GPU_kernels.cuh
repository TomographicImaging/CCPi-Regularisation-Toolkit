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

/* 
Raw CUDA Kernels for TV_PD regularisation model
*/

/************************************************/
/*****************2D modules*********************/
/************************************************/

extern "C" __global__ void dualPD_kernel(float *U, float *P1, float *P2, float sigma, int N, int M)
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
extern "C" __global__ void Proj_funcPD2D_iso_kernel(float *P1, float *P2, int N, int M)
{

   float denom;
   //calculate each thread global index
   const int xIndex=blockIdx.x*blockDim.x+threadIdx.x;
   const int yIndex=blockIdx.y*blockDim.y+threadIdx.y;

   int index = xIndex + N*yIndex;

   if ((xIndex < N) && (yIndex < M)) {
       denom = P1[index]*P1[index] +  P2[index]*P2[index];
       if (denom > 1.0f) {
           P1[index] = P1[index]/sqrtf(denom);
           P2[index] = P2[index]/sqrtf(denom);
       }
   }
   return;
}
extern "C" __global__ void Proj_funcPD2D_aniso_kernel(float *P1, float *P2, int N, int M)
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
extern "C" __global__ void DivProj2D_kernel(float *U, float *Input, float *P1, float *P2, float lt, float tau, int N, int M)
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
extern "C" __global__ void PDnonneg2D_kernel(float* Output, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);


   if (index < static_cast<long long>(dimX*dimY))	{
       if (Output[index] < 0.0f) Output[index] = 0.0f;
   }

}
/************************************************/
/*****************3D modules*********************/
/************************************************/
extern "C" __global__ void dualPD3D_kernel(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ)
{

    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;           
    
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

     if (i == dimX-1) {
     long long index1 = static_cast<long long>(i-1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);  
     P1[index] += sigma*(U[index1] - U[index]);
     }
     else {
     long long index2 = static_cast<long long>(i+1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);  
     P1[index] += sigma*(U[index2] - U[index]);
     }
     if (j == dimY-1) {
     long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j-1) + dimX * dimY * static_cast<long long>(k);
     P2[index] += sigma*(U[index3] - U[index]);
     }
     else {
    long long index4 = static_cast<long long>(i) + dimX * static_cast<long long>(j+1) + dimX * dimY * static_cast<long long>(k);        
    P2[index] += sigma*(U[index4] - U[index]);
     } 
     if (k == dimZ-1) {
    long long index5 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k-1);    
     P3[index] += sigma*(U[index5] - U[index]);
     }
     else  {
     long long index6 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k+1);
     P3[index] += sigma*(U[index6] - U[index]);
     }

   return;
}
extern "C" __global__ void Proj_funcPD3D_iso_kernel(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{

    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
           
   float denom,sq_denom;      
   long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

    denom = P1[index]*P1[index] +  P2[index]*P2[index] + P3[index]*P3[index];
       if (denom > 1.0f) {
           sq_denom = 1.0f/sqrtf(denom);
           P1[index] *= sq_denom;
           P2[index] *= sq_denom;
           P3[index] *= sq_denom;
       }
   return;
}

extern "C" __global__ void Proj_funcPD3D_aniso_kernel(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ)
{   
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;
    float val1, val2, val3;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

    val1 = abs(P1[index]);
    val2 = abs(P2[index]);
    val3 = abs(P3[index]);
    if (val1 < 1.0f) {val1 = 1.0f;}
    if (val2 < 1.0f) {val2 = 1.0f;}
    if (val3 < 1.0f) {val3 = 1.0f;}
    P1[index] /= val1;
    P2[index] /= val2;
    P3[index] /= val3;

   return;
}
extern "C" __global__ void DivProj3D_kernel(float *U, float *Input, float *P1, float *P2, float *P3, float lt, float tau, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;
    float P_v1, P_v2, P_v3, div_var;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);

    if (i == 0) P_v1 = -P1[index];
    else {
    long long index1 = static_cast<long long>(i-1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P_v1 = -(P1[index] - P1[index1]);
    }
    if (j == 0) P_v2 = -P2[index];
    else  {
    long long index2 = static_cast<long long>(i) + dimX * static_cast<long long>(j-1) + dimX * dimY * static_cast<long long>(k);
    P_v2 = -(P2[index] - P2[index2]);
    }
    if (k == 0) P_v3 = -P3[index];
    else  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k-1);
    P_v3 = -(P3[index] - P3[index3]);
    }
    
    div_var = P_v1 + P_v2 + P_v3;
    
    U[index] = (U[index] - tau*div_var + lt*Input[index])/(1.0f + lt);

   return;
}

extern "C" __global__ void PDnonneg3D_kernel(float* Output, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);


   if (index < static_cast<long long>(dimX*dimY*dimZ))	{
       if (Output[index] < 0.0f) Output[index] = 0.0f;
   }
}
extern "C" __global__ void PDcopy_kernel2D(float *Input, float* Output, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

   if (index < static_cast<long long>(dimX*dimY))	{
       Output[index] = Input[index];
   }
}

extern "C" __global__ void PDcopy_kernel3D(float *Input, float* Output, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);


   if (index < static_cast<long long>(dimX*dimY*dimZ))	{
       Output[index] = Input[index];
   }
}

extern "C" __global__ void getU2D_kernel(float *Input, float *Input_old, float theta, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

   if (index < static_cast<long long>(dimX*dimY))	{
       Input[index] += theta*(Input[index] - Input_old[index]);
   }
}

extern "C" __global__ void getU3D_kernel(float *Input, float *Input_old, float theta, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);


   if (index < static_cast<long long>(dimX*dimY*dimZ))	{
       Input[index] += theta*(Input[index] - Input_old[index]);
   }
}

extern "C" __global__ void PDResidCalc2D_kernel(float *Input1, float *Input2, float* Output, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j);

   if (index < static_cast<long long>(dimX*dimY))	{
       Output[index] = Input1[index] - Input2[index];
   }
}

extern "C" __global__ void PDResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;
    long long index = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);


   if (index < static_cast<long long>(dimX*dimY*dimZ))	
   {
        Output[index] = Input1[index] - Input2[index];
   }
}

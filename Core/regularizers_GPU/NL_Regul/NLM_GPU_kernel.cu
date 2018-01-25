#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "NLM_GPU_kernel.h"

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

extern __shared__ float sharedmem[];

// run PB den kernel here
__global__ void NLM_kernel(float *Ad, float* Bd, float *Eucl_Vec_d, int N, int M, int Z, int SearchW, int SimilW, int SearchW_real, int SearchW_full, int SimilW_full, int padXY, float h2, float lambda,  dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, int nUpdatePerThread, float neighborsize)
{
    
    int  i1, j1, k1, i2, j2, k2, i3, j3, k3, i_l, j_l, k_l, count;
    float value, Weight_norm, normsum, Weight;
    
    int bidx = blockIdx.x;
    int bidy = blockIdx.y%griddim.y;
    int bidz = (int)((blockIdx.y)/griddim.y);
    
    // global index for block endpoint
    int beidx = __mul24(bidx,blockDim.x);
    int beidy = __mul24(bidy,blockDim.y);
    int beidz = __mul24(bidz,blockDim.z);
    
    int tid = __mul24(threadIdx.z,__mul24(blockDim.x,blockDim.y)) +
            __mul24(threadIdx.y,blockDim.x) + threadIdx.x;
    
    #ifdef __DEVICE_EMULATION__
            printf("tid : %d", tid);
    #endif
            
    // update shared memory
    int nthreads = blockDim.x*blockDim.y*blockDim.z;
    int sharedMemSize = sharedmemdim.x * sharedmemdim.y * sharedmemdim.z;
    for(int i=0; i<nUpdatePerThread; i++)
    {
        int sid = tid + i*nthreads; // index in shared memory
        if (sid < sharedMemSize)
        {
            // global x/y/z index in volume
            int gidx, gidy, gidz;
            int sidx, sidy, sidz, tid;
            
            sidz = sid / (sharedmemdim.x*sharedmemdim.y);
            tid  = sid - sidz*(sharedmemdim.x*sharedmemdim.y);
            sidy = tid / (sharedmemdim.x);
            sidx = tid - sidy*(sharedmemdim.x);
            
            gidx = (int)sidx - (int)kerneldim.x + (int)beidx;
            gidy = (int)sidy - (int)kerneldim.y + (int)beidy;
            gidz = (int)sidz - (int)kerneldim.z + (int)beidz;
            
            // Neumann boundary condition
            int cx = (int) min(max(0,gidx),imagedim.x-1);
            int cy = (int) min(max(0,gidy),imagedim.y-1);
            int cz = (int) min(max(0,gidz),imagedim.z-1);
            
            int gid = cz*imagedim.x*imagedim.y + cy*imagedim.x + cx;
            
            sharedmem[sid] = Ad[gid];
        }
    }
    __syncthreads();
    
    // global index of the current voxel in the input volume
    int idx = beidx + threadIdx.x;
    int idy = beidy + threadIdx.y;
    int idz = beidz + threadIdx.z;
    
    if (Z == 1) {
        /* 2D case */
        /*checking boundaries to be within the image and avoid padded spaces */
        if( idx >= padXY && idx < (imagedim.x - padXY) &&
                idy >= padXY && idy < (imagedim.y - padXY))
        {
            int i_centr = threadIdx.x + (SearchW); /*indices of the centrilized (main) pixel */
            int j_centr = threadIdx.y + (SearchW); /*indices of the centrilized (main) pixel */
            
            if ((i_centr > 0) && (i_centr < N) && (j_centr > 0) && (j_centr < M)) {
                
                Weight_norm = 0; value = 0.0;
                /* Massive Search window loop */
                for(i1 = i_centr - SearchW_real ; i1 <= i_centr + SearchW_real; i1++) {
                    for(j1 = j_centr - SearchW_real ; j1<= j_centr + SearchW_real ; j1++) {
                        /* if inside the searching window */
                        count = 0; normsum = 0.0;
                        for(i_l=-SimilW; i_l<=SimilW; i_l++) {
                            for(j_l=-SimilW; j_l<=SimilW; j_l++) {
                                i2 = i1+i_l; j2 = j1+j_l;
                                i3 = i_centr+i_l; j3 = j_centr+j_l;  /*coordinates of the inner patch loop */                                
                                if ((i2 > 0) && (i2 < N) && (j2 > 0) && (j2 < M)) {
                                       if ((i3 > 0) && (i3 < N) && (j3 > 0) && (j3 < M)) {
                                            normsum += Eucl_Vec_d[count]*pow((sharedmem[(j3)*sharedmemdim.x+(i3)] - sharedmem[j2*sharedmemdim.x+i2]), 2);
                                            }}
                                        count++;
                                }}
                                if (normsum != 0) Weight = (expf(-normsum/h2));
                                else Weight = 0.0;
                                Weight_norm += Weight;
                                value += sharedmem[j1*sharedmemdim.x+i1]*Weight;
                            }}      
                                
                if (Weight_norm != 0) Bd[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = value/Weight_norm;
                else Bd[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = Ad[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx];
            }
        }      /*boundary conditions end*/
    }
    else {
        /*3D case*/
        /*checking boundaries to be within the image and avoid padded spaces */
        if( idx >= padXY && idx < (imagedim.x - padXY) &&
                idy >= padXY && idy < (imagedim.y - padXY) &&
                idz >= padXY && idz < (imagedim.z - padXY) )
        {
            int i_centr = threadIdx.x + SearchW; /*indices of the centrilized (main) pixel */
            int j_centr = threadIdx.y + SearchW; /*indices of the centrilized (main) pixel */
            int k_centr = threadIdx.z + SearchW; /*indices of the centrilized (main) pixel */
            
            if ((i_centr > 0) && (i_centr < N) && (j_centr > 0) && (j_centr < M) && (k_centr > 0) && (k_centr < Z)) {
                
                Weight_norm = 0; value = 0.0;
                /* Massive Search window loop */
                for(i1 = i_centr - SearchW_real ; i1 <= i_centr + SearchW_real; i1++) {
                    for(j1 = j_centr - SearchW_real ; j1<= j_centr + SearchW_real ; j1++) {
                        for(k1 = k_centr - SearchW_real ; k1<= k_centr + SearchW_real ; k1++) {
                            /* if inside the searching window */
                            count = 0; normsum = 0.0;
                            for(i_l=-SimilW; i_l<=SimilW; i_l++) {
                                for(j_l=-SimilW; j_l<=SimilW; j_l++) {
                                    for(k_l=-SimilW; k_l<=SimilW; k_l++) {
                                        i2 = i1+i_l; j2 = j1+j_l; k2 = k1+k_l;
                                        i3 = i_centr+i_l; j3 = j_centr+j_l; k3 = k_centr+k_l;   /*coordinates of the inner patch loop */                              
                                                    if ((i2 > 0) && (i2 < N) && (j2 > 0) && (j2 < M) && (k2 > 0) && (k2 < Z)) {
                                                        if ((i3 > 0) && (i3 < N) && (j3 > 0) && (j3 < M) && (k3 > 0) && (k3 < Z)) {
                                                            normsum += Eucl_Vec_d[count]*pow((sharedmem[(k3)*sharedmemdim.x*sharedmemdim.y + (j3)*sharedmemdim.x+(i3)] - sharedmem[(k2)*sharedmemdim.x*sharedmemdim.y + j2*sharedmemdim.x+i2]), 2);
                                                        }}
                                                    count++;
                                                }}}
                                       if (normsum != 0) Weight = (expf(-normsum/h2));
                                       else Weight = 0.0;
                                       Weight_norm += Weight;
                                       value += sharedmem[k1*sharedmemdim.x*sharedmemdim.y + j1*sharedmemdim.x+i1]*Weight;                                                                
                        }}}      /* BIG search window loop end*/
                
               
                if (Weight_norm != 0) Bd[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = value/Weight_norm;
                else Bd[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = Ad[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx];
            }
        }      /* boundary conditions end */
    }
}
        
/////////////////////////////////////////////////
// HOST FUNCTION
extern "C" void NLM_GPU_kernel(float *A, float* B, float *Eucl_Vec, int N, int M, int Z, int dimension, int SearchW, int SimilW, int SearchW_real, float h2, float lambda)
{
    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }
    
//     cudaDeviceReset();
    
    int padXY, SearchW_full, SimilW_full,  blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ, kernel_depth;
    float *Ad, *Bd, *Eucl_Vec_d;
    
    if (dimension == 2) {
        blockWidth  = 16;
        blockHeight = 16;
        blockDepth  = 1;
        Z = 1;
        kernel_depth = 0;
    }
    else {
        blockWidth  = 8;
        blockHeight = 8;
        blockDepth  = 8;
        kernel_depth = SearchW;
    }
    
    // compute how many blocks are needed
    nBlockX = ceil((float)N / (float)blockWidth);
    nBlockY = ceil((float)M / (float)blockHeight);
    nBlockZ = ceil((float)Z / (float)blockDepth);
    
    dim3 dimGrid(nBlockX,nBlockY*nBlockZ);
    dim3 dimBlock(blockWidth, blockHeight, blockDepth);
    dim3 imagedim(N,M,Z);
    dim3 griddim(nBlockX,nBlockY,nBlockZ);
    
    dim3 kerneldim(SearchW,SearchW,kernel_depth);
    dim3 sharedmemdim((SearchW*2)+blockWidth,(SearchW*2)+blockHeight,(kernel_depth*2)+blockDepth);
    int sharedmemsize = sizeof(float)*sharedmemdim.x*sharedmemdim.y*sharedmemdim.z;
    int updateperthread = ceil((float)(sharedmemdim.x*sharedmemdim.y*sharedmemdim.z)/(float)(blockWidth*blockHeight*blockDepth));
    float neighborsize = (2*SearchW+1)*(2*SearchW+1)*(2*kernel_depth+1);
    
    padXY = SearchW + 2*SimilW; /* padding sizes */
    
    SearchW_full = 2*SearchW + 1; /* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   /* the full similarity window  size */
    
    /*allocate space for images on device*/
    checkCudaErrors( cudaMalloc((void**)&Ad,N*M*Z*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&Bd,N*M*Z*sizeof(float)) );
    /*allocate space for vectors on device*/
    if (dimension == 2) {
        checkCudaErrors( cudaMalloc((void**)&Eucl_Vec_d,SimilW_full*SimilW_full*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(Eucl_Vec_d,Eucl_Vec,SimilW_full*SimilW_full*sizeof(float),cudaMemcpyHostToDevice) );
    }
    else {
        checkCudaErrors( cudaMalloc((void**)&Eucl_Vec_d,SimilW_full*SimilW_full*SimilW_full*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(Eucl_Vec_d,Eucl_Vec,SimilW_full*SimilW_full*SimilW_full*sizeof(float),cudaMemcpyHostToDevice) );
    }
    
    /* copy data from the host to device */
    checkCudaErrors( cudaMemcpy(Ad,A,N*M*Z*sizeof(float),cudaMemcpyHostToDevice) );
    
    // Run CUDA kernel here
    NLM_kernel<<<dimGrid,dimBlock,sharedmemsize>>>(Ad, Bd, Eucl_Vec_d, M, N, Z, SearchW, SimilW, SearchW_real, SearchW_full, SimilW_full, padXY, h2, lambda, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, neighborsize);
    
    checkCudaErrors( cudaPeekAtLastError() );
//     gpuErrchk( cudaDeviceSynchronize() );
    
    checkCudaErrors( cudaMemcpy(B,Bd,N*M*Z*sizeof(float),cudaMemcpyDeviceToHost) );
    cudaFree(Ad); cudaFree(Bd); cudaFree(Eucl_Vec_d);
}

float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY, int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop)
{
    /* padding-cropping function */
    int i,j,k;    
    if (NewSizeZ > 1) {    
           for (i=0; i < NewSizeX; i++) {
            for (j=0; j < NewSizeY; j++) {
              for (k=0; k < NewSizeZ; k++) {
                if (((i >= padXY) && (i < NewSizeX-padXY)) &&  ((j >= padXY) && (j < NewSizeY-padXY)) &&  ((k >= padXY) && (k < NewSizeZ-padXY))) {
                    if (switchpad_crop == 0)  Ap[NewSizeX*NewSizeY*k + i*NewSizeY+j] = A[OldSizeX*OldSizeY*(k - padXY) + (i-padXY)*(OldSizeY)+(j-padXY)];
                    else  Ap[OldSizeX*OldSizeY*(k - padXY) + (i-padXY)*(OldSizeY)+(j-padXY)] = A[NewSizeX*NewSizeY*k + i*NewSizeY+j];
                }
            }}}   
    }
    else {
        for (i=0; i < NewSizeX; i++) {
            for (j=0; j < NewSizeY; j++) {
                if (((i >= padXY) && (i < NewSizeX-padXY)) &&  ((j >= padXY) && (j < NewSizeY-padXY))) {
                    if (switchpad_crop == 0)  Ap[i*NewSizeY+j] = A[(i-padXY)*(OldSizeY)+(j-padXY)];
                    else  Ap[(i-padXY)*(OldSizeY)+(j-padXY)] = A[i*NewSizeY+j];
                }
            }}
    }
    return *Ap;
}
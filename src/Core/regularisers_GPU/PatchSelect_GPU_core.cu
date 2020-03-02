/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC and Diamond Light Source Ltd.
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 * Copyright 2018 Diamond Light Source Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PatchSelect_GPU_core.h"
#include "shared.h"

/* CUDA implementation of non-local weight pre-calculation for non-local priors
 * Weights and associated indices are stored into pre-allocated arrays and passed
 * to the regulariser
 *
 *
 * Input Parameters:
 * 1. 2D grayscale image (classical 3D version will not be supported but rather 2D + dim extension (TODO))
 * 2. Searching window (half-size of the main bigger searching window, e.g. 11)
 * 3. Similarity window (half-size of the patch window, e.g. 2)
 * 4. The number of neighbours to take (the most prominent after sorting neighbours will be taken)
 * 5. noise-related parameter to calculate non-local weights
 *
 * Output [2D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. Weights_ij - associated weights
 */


#define BLKXSIZE 8
#define BLKYSIZE 4
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )
#define M_PI 3.14159265358979323846
#define EPS 1.0e-8
#define CONSTVECSIZE5 121
#define CONSTVECSIZE7 225
#define CONSTVECSIZE9 361
#define CONSTVECSIZE11 529
#define CONSTVECSIZE13 729

__device__ void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}
__device__ void swapUS(unsigned short *xp, unsigned short *yp)
{
    unsigned short temp = *xp;
    *xp = *yp;
    *yp = temp;
}

/********************************************************************************/
__global__ void IndexSelect2D_5_kernel(float *Ad, unsigned short *H_i_d, unsigned short *H_j_d, float *Weights_d, float *Eucl_Vec_d, int N, int M, int SearchWindow, int SearchW_full, int SimilarWin, int NumNeighb, float h2)
{

    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, counterG, index2, ind;
    float normsum;

    float Weight_Vec[CONSTVECSIZE5];
    unsigned short ind_i[CONSTVECSIZE5];
    unsigned short ind_j[CONSTVECSIZE5];

    for(ind=0; ind<CONSTVECSIZE5; ind++) {
      Weight_Vec[ind] = 0.0;
      ind_i[ind] = 0;
      ind_j[ind] = 0; }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    long index = i + N*j;

    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
      i1 = i+i_m;
      if ((i1 >= 0) && (i1 < N)) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            j1 = j+j_m;
            if ((j1 >= 0) && (j1 < M)) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                  i2 = i1 + i_c;
                  i3 = i + i_c;
                  if ((i2 >= 0) && (i2 < N) && (i3 >= 0) && (i3 < N)) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        j2 = j1 + j_c;
                        j3 = j + j_c;
                        if ((j2 >= 0) && (j2 < M) && (j3 >= 0) && (j3 < M)) {
                                normsum += Eucl_Vec_d[counterG]*powf(Ad[i3 + N*j3] - Ad[i2 + N*j2], 2);
                                counterG++;
                              } /*if j2 j3*/
                          }
                    } /*if i2 i3*/
                   }
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
              } /*if j1*/
            }
          } /*if i1*/
        }

    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter-1; x++)  {
       for (y = 0; y < counter-x-1; y++)  {
           if (Weight_Vec[y] < Weight_Vec[y+1]) {
            swap(&Weight_Vec[y], &Weight_Vec[y+1]);
            swapUS(&ind_i[y], &ind_i[y+1]);
            swapUS(&ind_j[y], &ind_j[y+1]);
            }
    	}
    }
    /*sorting loop finished*/
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index2 = (N*M*x) + index;
        H_i_d[index2] = ind_i[x];
        H_j_d[index2] = ind_j[x];
        Weights_d[index2] = Weight_Vec[x];
    }
}
/********************************************************************************/
__global__ void IndexSelect2D_7_kernel(float *Ad, unsigned short *H_i_d, unsigned short *H_j_d, float *Weights_d, float *Eucl_Vec_d, int N, int M, int SearchWindow, int SearchW_full, int SimilarWin, int NumNeighb, float h2)
{

    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, counterG, index2, ind;
    float normsum;

    float Weight_Vec[CONSTVECSIZE7];
    unsigned short ind_i[CONSTVECSIZE7];
    unsigned short ind_j[CONSTVECSIZE7];

    for(ind=0; ind<CONSTVECSIZE7; ind++) {
      Weight_Vec[ind] = 0.0;
      ind_i[ind] = 0;
      ind_j[ind] = 0; }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    long index = i + N*j;

    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
      i1 = i+i_m;
      if ((i1 >= 0) && (i1 < N)) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            j1 = j+j_m;
            if ((j1 >= 0) && (j1 < M)) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                  i2 = i1 + i_c;
                  i3 = i + i_c;
                  if ((i2 >= 0) && (i2 < N) && (i3 >= 0) && (i3 < N)) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        j2 = j1 + j_c;
                        j3 = j + j_c;
                        if ((j2 >= 0) && (j2 < M) && (j3 >= 0) && (j3 < M)) {
                                normsum += Eucl_Vec_d[counterG]*powf(Ad[i3 + N*j3] - Ad[i2 + N*j2], 2);
                                counterG++;
                              } /*if j2 j3*/
                          }
                     } /*if i2 i3*/
                   }
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
              } /*if j1*/
            }
          } /*if i1*/
        }

    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter-1; x++)  {
       for (y = 0; y < counter-x-1; y++)  {
           if (Weight_Vec[y] < Weight_Vec[y+1]) {
            swap(&Weight_Vec[y], &Weight_Vec[y+1]);
            swapUS(&ind_i[y], &ind_i[y+1]);
            swapUS(&ind_j[y], &ind_j[y+1]);
            }
    	}
    }
    /*sorting loop finished*/
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index2 = (N*M*x) + index;
        H_i_d[index2] = ind_i[x];
        H_j_d[index2] = ind_j[x];
        Weights_d[index2] = Weight_Vec[x];
    }
}
__global__ void IndexSelect2D_9_kernel(float *Ad, unsigned short *H_i_d, unsigned short *H_j_d, float *Weights_d, float *Eucl_Vec_d, int N, int M, int SearchWindow, int SearchW_full, int SimilarWin, int NumNeighb, float h2)
{

    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, counterG, index2, ind;
    float normsum;

    float Weight_Vec[CONSTVECSIZE9];
    unsigned short ind_i[CONSTVECSIZE9];
    unsigned short ind_j[CONSTVECSIZE9];

    for(ind=0; ind<CONSTVECSIZE9; ind++) {
      Weight_Vec[ind] = 0.0;
      ind_i[ind] = 0;
      ind_j[ind] = 0; }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    long index = i + N*j;

    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
      i1 = i+i_m;
      if ((i1 >= 0) && (i1 < N)) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            j1 = j+j_m;
            if ((j1 >= 0) && (j1 < M)) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                  i2 = i1 + i_c;
                  i3 = i + i_c;
                  if ((i2 >= 0) && (i2 < N) && (i3 >= 0) && (i3 < N)) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        j2 = j1 + j_c;
                        j3 = j + j_c;
                        if ((j2 >= 0) && (j2 < M) && (j3 >= 0) && (j3 < M)) {
                                normsum += Eucl_Vec_d[counterG]*powf(Ad[i3 + N*j3] - Ad[i2 + N*j2], 2);
                                counterG++;
                              } /*if j2 j3*/
                          }
                    } /*if i2 i3*/
                   }
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
              } /*if j1*/
            }
          } /*if i1*/
        }

    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter-1; x++)  {
       for (y = 0; y < counter-x-1; y++)  {
           if (Weight_Vec[y] < Weight_Vec[y+1]) {
            swap(&Weight_Vec[y], &Weight_Vec[y+1]);
            swapUS(&ind_i[y], &ind_i[y+1]);
            swapUS(&ind_j[y], &ind_j[y+1]);
            }
    	}
    }
    /*sorting loop finished*/
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index2 = (N*M*x) + index;
        H_i_d[index2] = ind_i[x];
        H_j_d[index2] = ind_j[x];
        Weights_d[index2] = Weight_Vec[x];
    }
}
__global__ void IndexSelect2D_11_kernel(float *Ad, unsigned short *H_i_d, unsigned short *H_j_d, float *Weights_d, float *Eucl_Vec_d, int N, int M, int SearchWindow, int SearchW_full, int SimilarWin, int NumNeighb, float h2)
{

    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, counterG, index2, ind;
    float normsum;

    float Weight_Vec[CONSTVECSIZE11];
    unsigned short ind_i[CONSTVECSIZE11];
    unsigned short ind_j[CONSTVECSIZE11];

    for(ind=0; ind<CONSTVECSIZE11; ind++) {
      Weight_Vec[ind] = 0.0;
      ind_i[ind] = 0;
      ind_j[ind] = 0; }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    long index = i + N*j;

    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
      i1 = i+i_m;
      if ((i1 >= 0) && (i1 < N)) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            j1 = j+j_m;
            if ((j1 >= 0) && (j1 < M)) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                  i2 = i1 + i_c;
                  i3 = i + i_c;
                  if ((i2 >= 0) && (i2 < N) && (i3 >= 0) && (i3 < N)) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        j2 = j1 + j_c;
                        j3 = j + j_c;
                        if ((j2 >= 0) && (j2 < M) && (j3 >= 0) && (j3 < M)) {
                                normsum += Eucl_Vec_d[counterG]*powf(Ad[i3 + N*j3] - Ad[i2 + N*j2], 2);
                                counterG++;
                              } /*if j2 j3*/
                          }
                     } /*if i2 i3*/
                   }
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
              } /*if j1*/
            }
          } /*if i1*/
        }

    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter-1; x++)  {
       for (y = 0; y < counter-x-1; y++)  {
           if (Weight_Vec[y] < Weight_Vec[y+1]) {
            swap(&Weight_Vec[y], &Weight_Vec[y+1]);
            swapUS(&ind_i[y], &ind_i[y+1]);
            swapUS(&ind_j[y], &ind_j[y+1]);
            }
    	}
    }
    /*sorting loop finished*/
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index2 = (N*M*x) + index;
        H_i_d[index2] = ind_i[x];
        H_j_d[index2] = ind_j[x];
        Weights_d[index2] = Weight_Vec[x];
    }
}
__global__ void IndexSelect2D_13_kernel(float *Ad, unsigned short *H_i_d, unsigned short *H_j_d, float *Weights_d, float *Eucl_Vec_d, int N, int M, int SearchWindow, int SearchW_full, int SimilarWin, int NumNeighb, float h2)
{

    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, counterG, index2, ind;
    float normsum;

    float Weight_Vec[CONSTVECSIZE13];
    unsigned short ind_i[CONSTVECSIZE13];
    unsigned short ind_j[CONSTVECSIZE13];

    for(ind=0; ind<CONSTVECSIZE13; ind++) {
      Weight_Vec[ind] = 0.0;
      ind_i[ind] = 0;
      ind_j[ind] = 0; }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    long index = i + N*j;

    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
      i1 = i+i_m;
      if ((i1 >= 0) && (i1 < N)) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            j1 = j+j_m;
            if ((j1 >= 0) && (j1 < M)) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                  i2 = i1 + i_c;
                  i3 = i + i_c;
                  if ((i2 >= 0) && (i2 < N) && (i3 >= 0) && (i3 < N)) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        j2 = j1 + j_c;
                        j3 = j + j_c;
                        if ((j2 >= 0) && (j2 < M) && (j3 >= 0) && (j3 < M)) {
                                normsum += Eucl_Vec_d[counterG]*powf(Ad[i3 + N*j3] - Ad[i2 + N*j2], 2);
                                counterG++;
                              } /*if j2 j3*/
                          }
                     } /*if i2 i3*/
                   }
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
              } /*if j1*/
            }
          } /*if i1*/
        }

    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter-1; x++)  {
       for (y = 0; y < counter-x-1; y++)  {
           if (Weight_Vec[y] < Weight_Vec[y+1]) {
            swap(&Weight_Vec[y], &Weight_Vec[y+1]);
            swapUS(&ind_i[y], &ind_i[y+1]);
            swapUS(&ind_j[y], &ind_j[y+1]);
            }
    	}
    }
    /*sorting loop finished*/
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index2 = (N*M*x) + index;
        H_i_d[index2] = ind_i[x];
        H_j_d[index2] = ind_j[x];
        Weights_d[index2] = Weight_Vec[x];
    }
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/********************* MAIN HOST FUNCTION ******************/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
extern "C" int PatchSelect_GPU_main(float *A, unsigned short *H_i, unsigned short *H_j, float *Weights, int N, int M, int SearchWindow, int SimilarWin, int NumNeighb, float h)
{
    int deviceCount = -1; // number of devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    int SearchW_full, SimilW_full, counterG, i, j;
    float *Ad, *Weights_d, h2, *Eucl_Vec, *Eucl_Vec_d;
    unsigned short *H_i_d, *H_j_d;
    h2 = h*h;

    dim3 dimBlock(BLKXSIZE,BLKYSIZE);
    dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE));

    SearchW_full = (2*SearchWindow + 1)*(2*SearchWindow + 1); /* the full searching window  size */
    SimilW_full = (2*SimilarWin + 1)*(2*SimilarWin + 1);   /* the full similarity window  size */

    /* generate a 2D Gaussian kernel for NLM procedure */
    Eucl_Vec = (float*) calloc (SimilW_full,sizeof(float));
    counterG = 0;
    for(i=-SimilarWin; i<=SimilarWin; i++) {
         for(j=-SimilarWin; j<=SimilarWin; j++) {
              Eucl_Vec[counterG] = (float)exp(-(pow(((float) i), 2) + pow(((float) j), 2))/(2.0*SimilarWin*SimilarWin));
              counterG++;
    }} /*main neighb loop */


    /*allocate space on the device*/
    checkCudaErrors( cudaMalloc((void**)&Ad, N*M*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&H_i_d, N*M*NumNeighb*sizeof(unsigned short)) );
    checkCudaErrors( cudaMalloc((void**)&H_j_d, N*M*NumNeighb*sizeof(unsigned short)) );
    checkCudaErrors( cudaMalloc((void**)&Weights_d, N*M*NumNeighb*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&Eucl_Vec_d, SimilW_full*sizeof(float)) );

    /* copy data from the host to the device */
    checkCudaErrors( cudaMemcpy(Ad,A,N*M*sizeof(float),cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(Eucl_Vec_d,Eucl_Vec,SimilW_full*sizeof(float),cudaMemcpyHostToDevice) );

    /********************** Run CUDA kernel here ********************/
    if (SearchWindow == 5)  IndexSelect2D_5_kernel<<<dimGrid,dimBlock>>>(Ad, H_i_d, H_j_d, Weights_d, Eucl_Vec_d, N, M, SearchWindow, SearchW_full, SimilarWin, NumNeighb, h2);
    else if (SearchWindow == 7)  IndexSelect2D_7_kernel<<<dimGrid,dimBlock>>>(Ad, H_i_d, H_j_d, Weights_d, Eucl_Vec_d, N, M, SearchWindow, SearchW_full, SimilarWin, NumNeighb, h2);
    else if (SearchWindow == 9)  IndexSelect2D_9_kernel<<<dimGrid,dimBlock>>>(Ad, H_i_d, H_j_d, Weights_d, Eucl_Vec_d, N, M, SearchWindow, SearchW_full, SimilarWin, NumNeighb, h2);
    else if (SearchWindow == 11)  IndexSelect2D_11_kernel<<<dimGrid,dimBlock>>>(Ad, H_i_d, H_j_d, Weights_d, Eucl_Vec_d, N, M, SearchWindow, SearchW_full, SimilarWin, NumNeighb, h2);
    else if (SearchWindow == 13)  IndexSelect2D_13_kernel<<<dimGrid,dimBlock>>>(Ad, H_i_d, H_j_d, Weights_d, Eucl_Vec_d, N, M, SearchWindow, SearchW_full, SimilarWin, NumNeighb, h2);
    else {
    fprintf(stderr, "Select the searching window size from 5, 7, 9, 11 or 13\n");
        return -1;}
    checkCudaErrors(cudaPeekAtLastError() );
    checkCudaErrors(cudaDeviceSynchronize());
    /***************************************************************/

    checkCudaErrors(cudaMemcpy(H_i, H_i_d, N*M*NumNeighb*sizeof(unsigned short),cudaMemcpyDeviceToHost) );
    checkCudaErrors(cudaMemcpy(H_j, H_j_d, N*M*NumNeighb*sizeof(unsigned short),cudaMemcpyDeviceToHost) );
    checkCudaErrors(cudaMemcpy(Weights, Weights_d, N*M*NumNeighb*sizeof(float),cudaMemcpyDeviceToHost) );


    cudaFree(Ad);
    cudaFree(H_i_d);
    cudaFree(H_j_d);
    cudaFree(Weights_d);
    cudaFree(Eucl_Vec_d);
    cudaDeviceSynchronize();
    return 0;
}

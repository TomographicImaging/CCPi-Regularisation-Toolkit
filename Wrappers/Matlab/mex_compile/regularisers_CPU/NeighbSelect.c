/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
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

#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

#define EPS 1.0000e-12

/* C-OMP implementation of non-local weight pre-calculation for non-local priors
 * Weights and associated indices are stored into pre-allocated arrays and passed
 * to the regulariser
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. Searching window (half-size of the main bigger searching window, e.g. 11)
 * 3. Similarity window (half-size of the patch window, e.g. 2)
 * 4. The number of neighbours to take (the most prominent after sorting neighbours will be taken)
 * 5. noise-related parameter to calculate non-local weights
 *
 * Output [2D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. Weights_ij - associated weights
 *
 * Output [3D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. AR_k - indeces of j neighbours
 * 4. Weights_ijk - associated weights
 *
 * compile from Matlab with:
 * mex NeighbSearch2D_test.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 */

float Indeces2D(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimY, long dimX, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2);
float Indeces3D(float *Aorig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimY, long dimX, long dimZ, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2);

/**************************************************/
void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int number_of_dims,  SearchWindow, SimilarWin, NumNeighb, counterG;
    long i, j, k, dimX, dimY, dimZ;
    unsigned short *H_i=NULL, *H_j=NULL, *H_k=NULL;
    const int  *dim_array;
    float *A, *Weights, *Eucl_Vec, h, h2;
    int dim_array2[3]; /* for 2D data */
    int dim_array3[4]; /* for 3D data */
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /* a 2D or 3D image/volume */
    SearchWindow = (int) mxGetScalar(prhs[1]);    /* Large Searching window */
    SimilarWin = (int) mxGetScalar(prhs[2]);    /* Similarity window (patch-search)*/
    NumNeighb = (int) mxGetScalar(prhs[3]); /* the total number of neighbours to take */
    h = (float) mxGetScalar(prhs[4]); /* NLM parameter */
    h2 = h*h;
    
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    dim_array2[0] = dimX; dim_array2[1] = dimY; dim_array2[2] = NumNeighb;  /* 2D case */
    dim_array3[0] = dimX; dim_array3[1] = dimY; dim_array3[2] = dimZ; dim_array3[3] = NumNeighb;  /* 3D case */
    
    /****************2D INPUT ***************/
    if (number_of_dims == 2) {
        dimZ = 0;
        
        /* generate a 2D Gaussian kernel for NLM procedure */
        Eucl_Vec = (float*) calloc ((2*SimilarWin+1)*(2*SimilarWin+1),sizeof(float));
        counterG = 0;
        for(i=-SimilarWin; i<=SimilarWin; i++) {
            for(j=-SimilarWin; j<=SimilarWin; j++) {
                Eucl_Vec[counterG] = (float)exp(-(pow(((float) i), 2) + pow(((float) j), 2))/(2*SimilarWin*SimilarWin));
                counterG++;
            }} /*main neighb loop */
        
        H_i = (unsigned short*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        H_j = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        Weights = (float*)mxGetPr(plhs[2] = mxCreateNumericArray(3, dim_array2, mxSINGLE_CLASS, mxREAL));
        
        /* for each pixel store indeces of the most similar neighbours (patches) */
#pragma omp parallel for shared (A, Weights, H_i, H_j) private(i,j)
        for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {
                Indeces2D(A, H_i, H_j, Weights, i, j, dimX, dimY, Eucl_Vec, NumNeighb, SearchWindow, SimilarWin, h2);
            }}
    }
    /****************3D INPUT ***************/
    if (number_of_dims == 3) {
        
        /* generate a 3D Gaussian kernel for NLM procedure */
        Eucl_Vec = (float*) calloc ((2*SimilarWin+1)*(2*SimilarWin+1)*(2*SimilarWin+1),sizeof(float));
        counterG = 0;
        for(i=-SimilarWin; i<=SimilarWin; i++) {
            for(j=-SimilarWin; j<=SimilarWin; j++) {
                for(k=-SimilarWin; k<=SimilarWin; k++) {
                    Eucl_Vec[counterG] = (float)exp(-(pow(((float) i), 2) + pow(((float) j), 2) + pow(((float) k), 2))/(2*SimilarWin*SimilarWin*SimilarWin));
                    counterG++;
                }}} /*main neighb loop */
        
        H_i = (unsigned short*)mxGetPr(plhs[0] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        H_j = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        H_k = (unsigned short*)mxGetPr(plhs[2] = mxCreateNumericArray(4, dim_array3, mxUINT16_CLASS, mxREAL));
        Weights = (float*)mxGetPr(plhs[3] = mxCreateNumericArray(4, dim_array3, mxSINGLE_CLASS, mxREAL));
        
        /* for each voxel store indeces of the most similar neighbours (patches) */
#pragma omp parallel for shared (A, Weights, H_i, H_j, H_k) private(i,j,k)
        for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {
                for(k=0; k<dimZ; k++) {
                    Indeces3D(A, H_i, H_j, H_k, Weights, i, j, k, dimX, dimY, dimZ, Eucl_Vec, NumNeighb, SearchWindow, SimilarWin, h2);
                }}}
    }
    free(Eucl_Vec);
}

float Indeces2D(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimY, long dimX, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2)
{
    long i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, counter, x, y, index, sizeWin_tot, counterG;
    float *Weight_Vec, normsum, temp;
    unsigned short *ind_i, *ind_j, temp_i, temp_j;
    
    sizeWin_tot = (2*SearchWindow + 1)*(2*SearchWindow + 1);
    
    Weight_Vec = (float*) calloc(sizeWin_tot, sizeof(float));
    ind_i = (unsigned short*) calloc(sizeWin_tot, sizeof(unsigned short));
    ind_j = (unsigned short*) calloc(sizeWin_tot, sizeof(unsigned short));
    
    counter = 0;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                normsum = 0.0f; counterG = 0;
                for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                    for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                        i2 = i1 + i_c;
                        j2 = j1 + j_c;
                        i3 = i + i_c;
                        j3 = j + j_c;
                        if (((i2 >= 0) && (i2 < dimX)) && ((j2 >= 0) && (j2 < dimY))) {
                            if (((i3 >= 0) && (i3 < dimX)) && ((j3 >= 0) && (j3 < dimY))) {
                                normsum += Eucl_Vec[counterG]*pow(Aorig[j3*dimX + (i3)] - Aorig[j2*dimX + (i2)], 2);
                                counterG++;
                            }}
                        
                    }}
                /* writing temporarily into vectors */
                if (normsum > EPS) {
                    Weight_Vec[counter] = expf(-normsum/h2);
                    ind_i[counter] = i1;
                    ind_j[counter] = j1;
                    counter++;
                }
            }
        }}
    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter; x++)  {
        for (y = 0; y < counter; y++)  {
            if (Weight_Vec[y] < Weight_Vec[x]) {
                temp = Weight_Vec[y+1];
                temp_i = ind_i[y+1];
                temp_j = ind_j[y+1];
                Weight_Vec[y+1] = Weight_Vec[y];
                Weight_Vec[y] = temp;
                ind_i[y+1] = ind_i[y];
                ind_i[y] = temp_i;
                ind_j[y+1] = ind_j[y];
                ind_j[y] = temp_j;
            }}}
    /*sorting loop finished*/
    
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index = (dimX*dimY*x) + j*dimX+i;
        H_i[index] = ind_i[x];
        H_j[index] = ind_j[x];
        Weights[index] = Weight_Vec[x];
    }
    
    free(ind_i);
    free(ind_j);
    free(Weight_Vec);
    return 1;
}

float Indeces3D(float *Aorig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimY, long dimX, long dimZ, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2)
{
    long i1, j1, k1, i_m, j_m, k_m, i_c, j_c, k_c, i2, j2, k2, i3, j3, k3, counter, x, y, index, sizeWin_tot, counterG;
    float *Weight_Vec, normsum, temp;
    unsigned short *ind_i, *ind_j, *ind_k, temp_i, temp_j, temp_k;
    
    sizeWin_tot = (2*SearchWindow + 1)*(2*SearchWindow + 1)*(2*SearchWindow + 1);
    
    Weight_Vec = (float*) calloc(sizeWin_tot, sizeof(float));
    ind_i = (unsigned short*) calloc(sizeWin_tot, sizeof(unsigned short));
    ind_j = (unsigned short*) calloc(sizeWin_tot, sizeof(unsigned short));
    ind_k = (unsigned short*) calloc(sizeWin_tot, sizeof(unsigned short));
    
    counter = 0l;
    for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
        for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
            for(k_m=-SearchWindow; k_m<=SearchWindow; k_m++) {
                k1 = k+k_m;
                i1 = i+i_m;
                j1 = j+j_m;
                if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
                    normsum = 0.0f; counterG = 0l;
                    for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                        for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                            for(k_c=-SimilarWin; k_c<=SimilarWin; k_c++) {
                                i2 = i1 + i_c;
                                j2 = j1 + j_c;
                                k2 = k1 + k_c;
                                i3 = i + i_c;
                                j3 = j + j_c;
                                k3 = k + k_c;
                                if (((i2 >= 0) && (i2 < dimX)) && ((j2 >= 0) && (j2 < dimY)) && ((k2 >= 0) && (k2 < dimZ))) {
                                    if (((i3 >= 0) && (i3 < dimX)) && ((j3 >= 0) && (j3 < dimY)) && ((k3 >= 0) && (k3 < dimZ))) {
                                        normsum += Eucl_Vec[counterG]*pow(Aorig[(dimX*dimY*k3) + j3*dimX + (i3)] - Aorig[(dimX*dimY*k2) + j2*dimX + (i2)], 2);
                                        counterG++;
                                    }}
                            }}}
                    /* writing temporarily into vectors */
                    if (normsum > EPS) {
                        Weight_Vec[counter] = expf(-normsum/h2);
                        ind_i[counter] = i1;
                        ind_j[counter] = j1;
                        ind_k[counter] = k1;
                        counter ++;
                    }
                }
            }}}
    /* do sorting to choose the most prominent weights [HIGH to LOW] */
    /* and re-arrange indeces accordingly */
    for (x = 0; x < counter; x++)  {
        for (y = 0; y < counter; y++)  {
            if (Weight_Vec[y] < Weight_Vec[x]) {
                temp = Weight_Vec[y+1];
                temp_i = ind_i[y+1];
                temp_j = ind_j[y+1];
                temp_k = ind_k[y+1];
                Weight_Vec[y+1] = Weight_Vec[y];
                Weight_Vec[y] = temp;
                ind_i[y+1] = ind_i[y];
                ind_i[y] = temp_i;
                ind_j[y+1] = ind_j[y];
                ind_j[y] = temp_j;
                ind_k[y+1] = ind_k[y];
                ind_k[y] = temp_k;
            }}}
    /*sorting loop finished*/
    
    /*now select the NumNeighb more prominent weights and store into arrays */
    for(x=0; x < NumNeighb; x++) {
        index = dimX*dimY*dimZ*x + (dimX*dimY*k) + j*dimX+i;
        
        H_i[index] = ind_i[x];
        H_j[index] = ind_j[x];
        H_k[index] = ind_k[x];
        
        Weights[index] = Weight_Vec[x];
    }
    
    free(ind_i);
    free(ind_j);
    free(ind_k);
    free(Weight_Vec);
    return 1;
}


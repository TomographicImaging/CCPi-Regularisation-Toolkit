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

#define EPS 1.0000e-9

/* C-OMP implementation of non-local regulariser
 * Weights and associated indices must be given as an input 
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. AR_i - indeces of i neighbours
 * 3. AR_j - indeces of j neighbours
 * 4. AR_k - indeces of k neighbours (0 - for 2D case)
 * 5. Weights_ij(k) - associated weights 
 * 6. regularisation parameter
 * 7. iterations number 
 
 * Output:
 * 1. denoised image/volume 	
 * Elmoataz, Abderrahim, Olivier Lezoray, and Sébastien Bougleux. "Nonlocal discrete regularization on weighted graphs: a framework for image and manifold processing." IEEE Trans. Image Processing 17, no. 7 (2008): 1047-1060.
 
 */

float copyIm(float *A, float *U, long dimX, long dimY, long dimZ);
float NLM_H1_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float lambda);
float NLM_TV_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float lambda);
/**************************************************/

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    long number_of_dims,  i, j, k, dimX, dimY, dimZ, NumNeighb;
    int IterNumb, iter;
    unsigned short *H_i, *H_j, *H_k;
    const int  *dim_array;
    const int  *dim_array2;
    float *A_orig, *Output, *Weights, lambda;
    
    dim_array = mxGetDimensions(prhs[0]);
    dim_array2 = mxGetDimensions(prhs[1]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A_orig  = (float *) mxGetData(prhs[0]); /* a 2D image or a set of 2D images (3D stack) */
    H_i  = (unsigned short *) mxGetData(prhs[1]); /* indeces of i neighbours */
    H_j  = (unsigned short *) mxGetData(prhs[2]); /* indeces of j neighbours */
    H_k  = (unsigned short *) mxGetData(prhs[3]); /* indeces of k neighbours */
    Weights = (float *) mxGetData(prhs[4]); /* weights for patches */
    lambda = (float) mxGetScalar(prhs[5]); /* regularisation parameter */
    IterNumb = (int) mxGetScalar(prhs[6]); /* the number of iterations */
 
    lambda = 1.0f/lambda;       
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];   
         
    /*****2D INPUT *****/
    if (number_of_dims == 2) {
        dimZ = 0;
      
        NumNeighb = dim_array2[2];
        Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));        
        copyIm(A_orig, Output, (long)(dimX), (long)(dimY), 1l);
       
    /* for each pixel store indeces of the most similar neighbours (patches) */
     for(iter=0; iter<IterNumb; iter++) {    
#pragma omp parallel for shared (A_orig, Output, Weights, H_i, H_j, iter) private(i,j)
      for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {              
            //NLM_H1_2D(Output, A_orig, H_i, H_j, Weights, i, j, dimX, dimY, NumNeighb, lambda);
            NLM_TV_2D(Output, A_orig, H_i, H_j, Weights, i, j, dimX, dimY, NumNeighb, lambda);                              
           }}          
          }
    }
    /*****3D INPUT *****/
    /****************************************************/
    if (number_of_dims == 3) {
    Output = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));         
    }
}

/***********<<<<Main Function for ST NLM - H1 penalty>>>>**********/
float NLM_H1_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float lambda)
{
	long x, i1, j1, index; 
	float value = 0.0f, normweight  = 0.0f;
	
	for(x=0; x < NumNeighb; x++) {
	index =  (dimX*dimY*x) + j*dimX+i;
		i1 = H_i[index];
		j1 = H_j[index];
		value += A[j1*dimX+i1]*Weights[index];
		normweight += Weights[index];
	}	
    A[j*dimX+i] = (lambda*A_orig[j*dimX+i] + value)/(lambda + normweight);
    return *A;
}

/***********<<<<Main Function for ST NLM - TV penalty>>>>**********/
float NLM_TV_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float lambda)
{
	long x, i1, j1, index; 
	float value = 0.0f, normweight  = 0.0f, NLgrad_magn = 0.0f, NLCoeff;
	
	for(x=0; x < NumNeighb; x++) {
	index =  (dimX*dimY*x) + j*dimX+i;
		index =  (dimX*dimY*x) + j*dimX+i;
		i1 = H_i[index];
		j1 = H_j[index];
	        NLgrad_magn += powf((A[j1*dimX+i1] - A[j*dimX+i]),2)*Weights[index];
	}
  
    NLgrad_magn = sqrtf(NLgrad_magn); /*Non Local Gradients Magnitude */
    NLCoeff = 2.0f*(1.0f/(NLgrad_magn + EPS));
    		
    for(x=0; x < NumNeighb; x++) {
	index =  (dimX*dimY*x) + j*dimX+i;
	i1 = H_i[index];
	j1 = H_j[index];
        value += A[j1*dimX+i1]*NLCoeff*Weights[index];
        normweight += Weights[index]*NLCoeff;
    }   		
    A[j*dimX+i] = (lambda*A_orig[j*dimX+i] + value)/(lambda + normweight);
    return *A;
}



/* Copy Image (float) */
float copyIm(float *A, float *U, long dimX, long dimY, long dimZ)
{
	long j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return *U;
}



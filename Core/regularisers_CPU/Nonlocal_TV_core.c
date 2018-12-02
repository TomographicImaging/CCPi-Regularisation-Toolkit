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

#include "Nonlocal_TV_core.h"

/* C-OMP implementation of non-local regulariser
 * Weights and associated indices must be given as an input.
 * Gauss-Seidel fixed point iteration requires ~ 3 iterations, so the main effort
 * goes in pre-calculation of weights and selection of patches
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
/*****************************************************************************/

float Nonlocal_TV_CPU_main(float *A_orig, float *Output, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, int dimX, int dimY, int dimZ, int NumNeighb, float lambdaReg, int IterNumb)
{

    long i, j, k;
    int iter;
    lambdaReg = 1.0f/lambdaReg;
         
    /*****2D INPUT *****/
    if (dimZ == 0) {
	  copyIm(A_orig, Output, (long)(dimX), (long)(dimY), 1l);
    /* for each pixel store indeces of the most similar neighbours (patches) */
     for(iter=0; iter<IterNumb; iter++) {    
#pragma omp parallel for shared (A_orig, Output, Weights, H_i, H_j, iter) private(i,j)
      for(i=0; i<(long)(dimX); i++) {
            for(j=0; j<(long)(dimY); j++) {              
             /*NLM_H1_2D(Output, A_orig, H_i, H_j, Weights, i, j, (long)(dimX), (long)(dimY), NumNeighb, lambdaReg);*/  /* NLM - H1 penalty */
             NLM_TV_2D(Output, A_orig, H_i, H_j, Weights, i, j, (long)(dimX), (long)(dimY), NumNeighb, lambdaReg);  /* NLM - TV penalty */
           }}
          }
    }  
    else {
     /*****3D INPUT *****/
        copyIm(A_orig, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    /* for each pixel store indeces of the most similar neighbours (patches) */
     for(iter=0; iter<IterNumb; iter++) {    
#pragma omp parallel for shared (A_orig, Output, Weights, H_i, H_j, H_k, iter) private(i,j,k)
      for(i=0; i<(long)(dimX); i++) {
            for(j=0; j<(long)(dimY); j++) {              
               for(k=0; k<(long)(dimZ); k++) {
            /* NLM_H1_3D(Output, A_orig, H_i, H_j, H_k, Weights, i, j, k, dimX, dimY, dimZ, NumNeighb, lambdaReg); */ /* NLM - H1 penalty */
            NLM_TV_3D(Output, A_orig, H_i, H_j, H_k, Weights, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ), NumNeighb, lambdaReg);   /* NLM - TV penalty */     
           }}}          
          }          
    }
    return *Output;
}

/***********<<<<Main Function for NLM - H1 penalty>>>>**********/
float NLM_H1_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, int NumNeighb, float lambdaReg)
{
	long x, i1, j1, index, index_m; 
	float value = 0.0f, normweight  = 0.0f;
	
	index_m = j*dimX+i;
	for(x=0; x < NumNeighb; x++) {
	index =  (dimX*dimY*x) + j*dimX+i;
		i1 = H_i[index];
		j1 = H_j[index];
		value += A[j1*dimX+i1]*Weights[index];
		normweight += Weights[index];
	}
	 A[index_m] = (lambdaReg*A_orig[index_m] + value)/(lambdaReg + normweight);
    return *A;
}
/*3D version*/
float NLM_H1_3D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimX, long dimY, long dimZ, int NumNeighb, float lambdaReg)
{
	long x, i1, j1, k1, index; 
	float value = 0.0f, normweight  = 0.0f;
	
	for(x=0; x < NumNeighb; x++) {
	index = dimX*dimY*dimZ*x + (dimX*dimY*k) + j*dimX+i;
		i1 = H_i[index];
		j1 = H_j[index];
		k1 = H_k[index];
		value += A[(dimX*dimY*k1) + j1*dimX+i1]*Weights[index];
		normweight += Weights[index];
	}	
    A[(dimX*dimY*k) + j*dimX+i] = (lambdaReg*A_orig[(dimX*dimY*k) + j*dimX+i] + value)/(lambdaReg + normweight);
    return *A;
}


/***********<<<<Main Function for NLM - TV penalty>>>>**********/
float NLM_TV_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, int NumNeighb, float lambdaReg)
{
	long x, i1, j1, index, index_m; 
	float value = 0.0f, normweight  = 0.0f, NLgrad_magn = 0.0f, NLCoeff;
	
	 index_m = j*dimX+i;
		
	for(x=0; x < NumNeighb; x++) {
		index =  (dimX*dimY*x) + j*dimX+i; /*c*/
		i1 = H_i[index];
		j1 = H_j[index];
		NLgrad_magn += powf((A[j1*dimX+i1] - A[index_m]),2)*Weights[index];
	}
  
    NLgrad_magn = sqrtf(NLgrad_magn); /*Non Local Gradients Magnitude */
    NLCoeff = 2.0f*(1.0f/(NLgrad_magn + EPS));
    		
    for(x=0; x < NumNeighb; x++) {
	index =  (dimX*dimY*x) + j*dimX+i; /*c*/
	i1 = H_i[index];
	j1 = H_j[index];
        value += A[j1*dimX+i1]*NLCoeff*Weights[index];
        normweight += Weights[index]*NLCoeff;
    }   		
    A[index_m] = (lambdaReg*A_orig[index_m] + value)/(lambdaReg + normweight);
    return *A;
}
/*3D version*/
float NLM_TV_3D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimX, long dimY, long dimZ, int NumNeighb, float lambdaReg)
{
	long x, i1, j1, k1, index; 
	float value = 0.0f, normweight  = 0.0f, NLgrad_magn = 0.0f, NLCoeff;
	
	for(x=0; x < NumNeighb; x++) {
	index =  dimX*dimY*dimZ*x + (dimX*dimY*k) + j*dimX+i;
		i1 = H_i[index];
		j1 = H_j[index];
		k1 = H_k[index];
	        NLgrad_magn += powf((A[(dimX*dimY*k1) + j1*dimX+i1] - A[(dimX*dimY*k1) + j*dimX+i]),2)*Weights[index];
	}
  
    NLgrad_magn = sqrtf(NLgrad_magn); /*Non Local Gradients Magnitude */
    NLCoeff = 2.0f*(1.0f/(NLgrad_magn + EPS));
    		
    for(x=0; x < NumNeighb; x++) {
	index = dimX*dimY*dimZ*x + (dimX*dimY*k) + j*dimX+i;
	i1 = H_i[index];
	j1 = H_j[index];
	k1 = H_k[index];
        value += A[(dimX*dimY*k1) + j1*dimX+i1]*NLCoeff*Weights[index];
        normweight += Weights[index]*NLCoeff;
    }   		
    A[(dimX*dimY*k) + j*dimX+i] = (lambdaReg*A_orig[(dimX*dimY*k) + j*dimX+i] + value)/(lambdaReg + normweight);
    return *A;
}

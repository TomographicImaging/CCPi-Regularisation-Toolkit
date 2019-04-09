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

#include "DiffusionMASK_core.h"
#include "utils.h"

#define EPS 1.0e-5
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*sign function*/
int signNDF_m(float x) {
    return (x > 0) - (x < 0);
}

/* C-OMP implementation of linear and nonlinear diffusion [1,2] which is constrained by the provided MASK.
 * The minimisation is performed using explicit scheme.
 * Implementation using the diffusivity window to increase the coverage area of the diffusivity
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. MASK (in unsigned char format)
 * 3. Diffusivity window (half-size of the searching window, e.g. 3) 	 
 * 4. lambda - regularization parameter
 * 5. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 6. Number of iterations, for explicit scheme >= 150 is recommended
 * 7. tau - time-marching step for explicit scheme
 * 8. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
 * 9. eplsilon - tolerance constant

 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 * [2] Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.
 */

float DiffusionMASK_CPU_main(float *Input, unsigned char *MASK, float *Output, float *infovector, int DiffusWindow, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, float epsil, int dimX, int dimY, int dimZ)
{
    int i,j,k,counterG;
    float sigmaPar2, *Output_prev=NULL, *Eucl_Vec;
    int DiffusWindow_tot;
    sigmaPar2 = sigmaPar/sqrt(2.0f);
    long DimTotal;
    float re, re1;
    re = 0.0f; re1 = 0.0f;
    int count = 0;
    DimTotal = (long)(dimX*dimY*dimZ);

    if (dimZ == 1) {
	DiffusWindow_tot = (2*DiffusWindow + 1)*(2*DiffusWindow + 1);
        /* generate a 2D Gaussian kernel for NLM procedure */
        Eucl_Vec = (float*) calloc (DiffusWindow_tot,sizeof(float));
        counterG = 0;
        for(i=-DiffusWindow; i<=DiffusWindow; i++) {
            for(j=-DiffusWindow; j<=DiffusWindow; j++) {
                Eucl_Vec[counterG] = (float)expf(-(powf(((float) i), 2) + powf(((float) j), 2))/(2.0f*DiffusWindow*DiffusWindow));
                counterG++;
            }} /*main neighb loop */
       }
    else {
	DiffusWindow_tot = (2*DiffusWindow + 1)*(2*DiffusWindow + 1)*(2*DiffusWindow + 1);
	Eucl_Vec = (float*) calloc (DiffusWindow_tot,sizeof(float));
        counterG = 0;
        for(i=-DiffusWindow; i<=DiffusWindow; i++) {
            for(j=-DiffusWindow; j<=DiffusWindow; j++) {
                for(k=-DiffusWindow; k<=DiffusWindow; k++) {
                    Eucl_Vec[counterG] = (float)expf(-(powf(((float) i), 2) + powf(((float) j), 2) + powf(((float) k), 2))/(2*DiffusWindow*DiffusWindow*DiffusWindow));
                    counterG++;
                }}} /*main neighb loop */            
    }

    if (epsil != 0.0f) Output_prev = calloc(DimTotal, sizeof(float));

    /* copy into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    for(i=0; i < iterationsNumb; i++) {

      if ((epsil != 0.0f)  && (i % 5 == 0)) copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
      if (dimZ == 1) {
	    /* running 2D diffusion iterations */
            if (sigmaPar == 0.0f) LinearDiff_MASK2D(Input, MASK, Output, Eucl_Vec, DiffusWindow, lambdaPar, tau, (long)(dimX), (long)(dimY)); /* constrained linear diffusion */
            else NonLinearDiff_MASK2D(Input, MASK, Output, Eucl_Vec, DiffusWindow, lambdaPar, sigmaPar2, tau, penaltytype, (long)(dimX), (long)(dimY)); /* constrained nonlinear diffusion */
          }
      else {
       	/* running 3D diffusion iterations */
        //if (sigmaPar == 0.0f) LinearDiff3D(Input, Output, lambdaPar, tau, (long)(dimX), (long)(dimY), (long)(dimZ));
//       else NonLinearDiff3D(Input, Output, lambdaPar, sigmaPar2, tau, penaltytype, (long)(dimX), (long)(dimY), (long)(dimZ));
          }
          /* check early stopping criteria if epsilon not equal zero */
          if ((epsil != 0.0f)  && (i % 5 == 0)) {
          re = 0.0f; re1 = 0.0f;
            for(j=0; j<DimTotal; j++)
            {
                re += powf(Output[j] - Output_prev[j],2);
                re1 += powf(Output[j],2);
            }
          re = sqrtf(re)/sqrtf(re1);
          /* stop if the norm residual is less than the tolerance EPS */
          if (re < epsil)  count++;
          if (count > 3) break;
          }
		}

    free(Output_prev);
  /*adding info into info_vector */
    infovector[0] = (float)(i);  /*iterations number (if stopped earlier based on tolerance)*/
    infovector[1] = re;  /* reached tolerance */
    return 0;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/* MASKED-constrained 2D linear diffusion (PDE heat equation) */
float LinearDiff_MASK2D(float *Input, unsigned char *MASK, float *Output,  float *Eucl_Vec, int DiffusWindow, float lambdaPar, float tau, long dimX, long dimY)
{

long i,j,i1,j1,i_m,j_m,index,indexneighb,counter;
unsigned char class_c, class_n;
float diffVal;

#pragma omp parallel for shared(Input) private(index,i,j,i1,j1,i_m,j_m,counter,diffVal,indexneighb,class_c,class_n)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {        
        index = j*dimX+i; /* current pixel index */
        counter = 0; diffVal = 0.0f;
        for(i_m=-DiffusWindow; i_m<=DiffusWindow; i_m++) {
            for(j_m=-DiffusWindow; j_m<=DiffusWindow; j_m++) {		
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {                   
            indexneighb = j1*dimX+i1; /* neighbour pixel index */
	    class_c = MASK[index]; /* current class value */
    	    class_n = MASK[indexneighb]; /* neighbour class value */
	    
	    /* perform diffusion only within the same class (given by MASK) */
	    if (class_n == class_c) diffVal += Output[indexneighb] - Output[index];
            }
		counter++;
	    }}
            Output[index] += tau*(lambdaPar*(diffVal) - (Output[index] - Input[index]));
        }}
	return *Output;
}

/*  MASKED-constrained 2D nonlinear diffusion */
float NonLinearDiff_MASK2D(float *Input, unsigned char *MASK, float *Output, float *Eucl_Vec, int DiffusWindow, float lambdaPar, float sigmaPar, float tau, int penaltytype, long dimX, long dimY)
{
	long i,j,i1,j1,i_m,j_m,index,indexneighb,counter;
	unsigned char class_c, class_n;
	float diffVal, funcVal;
	
#pragma omp parallel for shared(Input) private(index,i,j,i1,j1,i_m,j_m,counter,diffVal,funcVal,indexneighb,class_c,class_n)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {        
        index = j*dimX+i; /* current pixel index */
        counter = 0; diffVal = 0.0f; funcVal = 0.0f;
        for(i_m=-DiffusWindow; i_m<=DiffusWindow; i_m++) {
            for(j_m=-DiffusWindow; j_m<=DiffusWindow; j_m++) {		
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {                   
            indexneighb = j1*dimX+i1; /* neighbour pixel index */
	    class_c = MASK[index]; /* current class value */
    	    class_n = MASK[indexneighb]; /* neighbour class value */
	    
	    /* perform diffusion only within the same class (given by MASK) */
	    if (class_n == class_c) {
	    	diffVal = Output[indexneighb] - Output[index];
	    	if (penaltytype == 1) {
	        /* Huber penalty */
                if (fabs(diffVal) > sigmaPar) funcVal += signNDF_m(diffVal);
                else funcVal += diffVal/sigmaPar; }
  		else if (penaltytype == 2) {
  		/* Perona-Malik */
  		funcVal += (diffVal)/(1.0f + powf((diffVal/sigmaPar),2)); }
  		else if (penaltytype == 3) {
  		/* Tukey Biweight */
  		if (fabs(diffVal) <= sigmaPar) funcVal += diffVal*powf((1.0f - powf((diffVal/sigmaPar),2)), 2); }
                else {
                printf("%s \n", "No penalty function selected! Use Huber,2 or 3.");
		break; }  		  
            		}
            	}
		counter++;
	    }}
           Output[index] += tau*(lambdaPar*(funcVal) - (Output[index] - Input[index]));
		}}
	return *Output;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/

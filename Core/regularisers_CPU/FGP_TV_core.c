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

#include "FGP_TV_core.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume 
 * 2. lambdaPar - regularization parameter 
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant 
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default) 
 * 7. print information: 0 (off) or 1 (on) 
 *
 * Output:
 * [1] Filtered/regularized image
 *
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 */
 
float TV_FGP_CPU_main(float *Input, float *Output, float lambdaPar, int iterationsNumb, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ)
{
	int ll;
    long j, DimTotal;
	float re, re1;
	float tk = 1.0f;
    float tkp1=1.0f;
    int count = 0;
	
	if (dimZ <= 1) {
		/*2D case */
		float *Output_prev=NULL, *P1=NULL, *P2=NULL, *P1_prev=NULL, *P2_prev=NULL, *R1=NULL, *R2=NULL;
		DimTotal = (long)(dimX*dimY);
		
        Output_prev = calloc(DimTotal, sizeof(float));
        P1 = calloc(DimTotal, sizeof(float));
        P2 = calloc(DimTotal, sizeof(float));
        P1_prev = calloc(DimTotal, sizeof(float));
        P2_prev = calloc(DimTotal, sizeof(float));
        R1 = calloc(DimTotal, sizeof(float));
        R2 = calloc(DimTotal, sizeof(float)); 
		
		/* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {
            
            /* computing the gradient of the objective function */
            Obj_func2D(Input, Output, R1, R2, lambdaPar, (long)(dimX), (long)(dimY));
            
            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (Output[j] < 0.0f) Output[j] = 0.0f;}
            
            /*Taking a step towards minus of the gradient*/
            Grad_func2D(P1, P2, Output, R1, R2, lambdaPar, (long)(dimX), (long)(dimY));
            
            /* projection step */
            Proj_func2D(P1, P2, methodTV, DimTotal);
            
            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func2D(P1, P1_prev, P2, P2_prev, R1, R2, tkp1, tk, DimTotal);
            
            /* check early stopping criteria */
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<DimTotal; j++)
            {
                re += pow(Output[j] - Output_prev[j],2);
                re1 += pow(Output[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            if (re < epsil)  count++;
				if (count > 4) break;
            
            /*storing old values*/
            copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), 1l);
            copyIm(P1, P1_prev, (long)(dimX), (long)(dimY), 1l);
            copyIm(P2, P2_prev, (long)(dimX), (long)(dimY), 1l);
            tk = tkp1;
        }
        if (printM == 1) printf("FGP-TV iterations stopped at iteration %i \n", ll);   
		free(Output_prev); free(P1); free(P2); free(P1_prev); free(P2_prev); free(R1); free(R2);		
	}
	else {
		/*3D case*/
		float *Output_prev=NULL, *P1=NULL, *P2=NULL, *P3=NULL, *P1_prev=NULL, *P2_prev=NULL, *P3_prev=NULL, *R1=NULL, *R2=NULL, *R3=NULL;		
		DimTotal = (long)(dimX*dimY*dimZ);        
        
        Output_prev = calloc(DimTotal, sizeof(float));
        P1 = calloc(DimTotal, sizeof(float));
        P2 = calloc(DimTotal, sizeof(float));
        P3 = calloc(DimTotal, sizeof(float));
        P1_prev = calloc(DimTotal, sizeof(float));
        P2_prev = calloc(DimTotal, sizeof(float));        
        P3_prev = calloc(DimTotal, sizeof(float));        
        R1 = calloc(DimTotal, sizeof(float));
        R2 = calloc(DimTotal, sizeof(float)); 
        R3 = calloc(DimTotal, sizeof(float)); 
		
		    /* begin iterations */
        for(ll=0; ll<iterationsNumb; ll++) {
            
            /* computing the gradient of the objective function */
            Obj_func3D(Input, Output, R1, R2, R3, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /* apply nonnegativity */
            if (nonneg == 1) for(j=0; j<DimTotal; j++) {if (Output[j] < 0.0f) Output[j] = 0.0f;}  
            
            /*Taking a step towards minus of the gradient*/
            Grad_func3D(P1, P2, P3, Output, R1, R2, R3, lambdaPar, (long)(dimX), (long)(dimY), (long)(dimZ));
            
            /* projection step */
            Proj_func3D(P1, P2, P3, methodTV, DimTotal);
            
            /*updating R and t*/
            tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
            Rupd_func3D(P1, P1_prev, P2, P2_prev, P3, P3_prev, R1, R2, R3, tkp1, tk, DimTotal);
            
            /* calculate norm - stopping rules*/
            re = 0.0f; re1 = 0.0f;
            for(j=0; j<DimTotal; j++)
            {
                re += pow(Output[j] - Output_prev[j],2);
                re1 += pow(Output[j],2);
            }
            re = sqrt(re)/sqrt(re1);
            /* stop if the norm residual is less than the tolerance EPS */
            if (re < epsil)  count++;
            if (count > 4) break;            
                        
            /*storing old values*/
            copyIm(Output, Output_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            copyIm(P1, P1_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            copyIm(P2, P2_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            copyIm(P3, P3_prev, (long)(dimX), (long)(dimY), (long)(dimZ));
            tk = tkp1;            
        }	
		if (printM == 1) printf("FGP-TV iterations stopped at iteration %i \n", ll);   
		free(Output_prev); free(P1); free(P2); free(P3); free(P1_prev); free(P2_prev); free(P3_prev); free(R1); free(R2); free(R3);
	}
	return *Output;
}

float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, long dimX, long dimY)
{
    float val1, val2;
    long i,j,index;
#pragma omp parallel for shared(A,D,R1,R2) private(index,i,j,val1,val2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            /* boundary conditions  */
            if (i == 0) {val1 = 0.0f;} else {val1 = R1[j*dimX + (i-1)];}
            if (j == 0) {val2 = 0.0f;} else {val2 = R2[(j-1)*dimX + i];}
            D[index] = A[index] - lambda*(R1[index] + R2[index] - val1 - val2);
        }}
    return *D;
}
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda,  long dimX, long dimY)
{
    float val1, val2, multip;
    long i,j,index;
    multip = (1.0f/(8.0f*lambda));
#pragma omp parallel for shared(P1,P2,D,R1,R2,multip) private(index,i,j,val1,val2)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
			index = j*dimX+i;
            /* boundary conditions */
            if (i == dimX-1) val1 = 0.0f; else val1 = D[index] - D[j*dimX + (i+1)];
            if (j == dimY-1) val2 = 0.0f; else val2 = D[index] - D[(j+1)*dimX + i];
            P1[index] = R1[index] + multip*val1;
            P2[index] = R2[index] + multip*val2;
        }}
    return 1;
}
float Proj_func2D(float *P1, float *P2, int methTV, long DimTotal)
{
    float val1, val2, denom, sq_denom;
    long i;
    if (methTV == 0) {
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,denom,sq_denom)
        for(i=0; i<DimTotal; i++) {
                denom = powf(P1[i],2) +  powf(P2[i],2);
                if (denom > 1.0f) {
					sq_denom = 1.0f/sqrtf(denom);
                    P1[i] = P1[i]*sq_denom;
                    P2[i] = P2[i]*sq_denom;
                }
            }
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,val1,val2)
        for(i=0; i<DimTotal; i++) {
                val1 = fabs(P1[i]);
                val2 = fabs(P2[i]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                P1[i] = P1[i]/val1;
                P2[i] = P2[i]/val2;
            }
    }
    return 1;
}
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, long DimTotal)
{
    long i;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P1_old,P2_old,R1,R2,multip) private(i)
    for(i=0; i<DimTotal; i++) {       
            R1[i] = P1[i] + multip*(P1[i] - P1_old[i]);
            R2[i] = P2[i] + multip*(P2[i] - P2_old[i]);
        }
    return 1;
}

/* 3D-case related Functions */
/*****************************************************************/
float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, long dimX, long dimY, long dimZ)
{
    float val1, val2, val3;
    long i,j,k,index;
#pragma omp parallel for shared(A,D,R1,R2,R3) private(index,i,j,k,val1,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
				index = (dimX*dimY)*k + j*dimX+i;
                /* boundary conditions */
                if (i == 0) {val1 = 0.0f;} else {val1 = R1[(dimX*dimY)*k + j*dimX + (i-1)];}
                if (j == 0) {val2 = 0.0f;} else {val2 = R2[(dimX*dimY)*k + (j-1)*dimX + i];}
                if (k == 0) {val3 = 0.0f;} else {val3 = R3[(dimX*dimY)*(k-1) + j*dimX + i];}
                D[index] = A[index] - lambda*(R1[index] + R2[index] + R3[index] - val1 - val2 - val3);
            }}}
    return *D;
}
float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float lambda, long dimX, long dimY, long dimZ)
{
    float val1, val2, val3, multip;
    long i,j,k, index;
    multip = (1.0f/(26.0f*lambda));
#pragma omp parallel for shared(P1,P2,P3,D,R1,R2,R3,multip) private(index,i,j,k,val1,val2,val3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
				index = (dimX*dimY)*k + j*dimX+i;				
                /* boundary conditions */
                if (i == dimX-1) val1 = 0.0f; else val1 = D[index] - D[(dimX*dimY)*k + j*dimX + (i+1)];
                if (j == dimY-1) val2 = 0.0f; else val2 = D[index] - D[(dimX*dimY)*k + (j+1)*dimX + i];
                if (k == dimZ-1) val3 = 0.0f; else val3 = D[index] - D[(dimX*dimY)*(k+1) + j*dimX + i];
                P1[index] = R1[index] + multip*val1;
                P2[index] = R2[index] + multip*val2;
                P3[index] = R3[index] + multip*val3;
            }}}
    return 1;
}
float Proj_func3D(float *P1, float *P2, float *P3, int methTV, long DimTotal)
{		
    float val1, val2, val3, denom, sq_denom;
    long i;
    if (methTV == 0) {
	/* isotropic TV*/
	#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3,sq_denom)
    for(i=0; i<DimTotal; i++) {        
				denom = powf(P1[i],2) + powf(P2[i],2) + powf(P3[i],2);
                if (denom > 1.0f) {
					sq_denom = 1.0f/sqrtf(denom);
                    P1[i] = P1[i]*sq_denom;
                    P2[i] = P2[i]*sq_denom;
                    P3[i] = P3[i]*sq_denom;
                }
			}
	}    
    else {
    /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3)
    for(i=0; i<DimTotal; i++) {
                val1 = fabs(P1[i]);
                val2 = fabs(P2[i]);
                val3 = fabs(P3[i]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                if (val3 < 1.0f) {val3 = 1.0f;}                
                P1[i] = P1[i]/val1;
                P2[i] = P2[i]/val2;
                P3[i] = P3[i]/val3;
            }
		}
    return 1;
}
float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, long DimTotal)
{
    long i;
    float multip;
    multip = ((tk-1.0f)/tkp1);
#pragma omp parallel for shared(P1,P2,P3,P1_old,P2_old,P3_old,R1,R2,R3,multip) private(i)
    for(i=0; i<DimTotal; i++) {
                R1[i] = P1[i] + multip*(P1[i] - P1_old[i]);
                R2[i] = P2[i] + multip*(P2[i] - P2_old[i]);
                R3[i] = P3[i] + multip*(P3[i] - P3_old[i]);
            }
    return 1;
}

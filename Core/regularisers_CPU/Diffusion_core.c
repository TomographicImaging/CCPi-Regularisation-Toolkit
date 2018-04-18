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

#include "Diffusion_core.h"
#include "utils.h"

#define EPS 1.0e-5
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*sign function*/
int sign(float x) {
    return (x > 0) - (x < 0);
}

/* C-OMP implementation of linear and nonlinear diffusion with the regularisation model [1,2] (2D/3D case)
 * The minimisation is performed using explicit scheme. 
 *
 * Input Parameters:
 * 1. Noisy image/volume 
 * 2. lambda - regularization parameter
 * 3. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 4. Number of iterations, for explicit scheme >= 150 is recommended 
 * 5. tau - time-marching step for explicit scheme
 * 6. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
 *
 * Output:
 * [1] Regularized image/volume 
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 * [2] Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.
 */

float Diffusion_CPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int dimX, int dimY, int dimZ)
{
    int i;
    float sigmaPar2;
    sigmaPar2 = sigmaPar/sqrt(2.0f);
    
    /* copy into output */
    copyIm(Input, Output, dimX, dimY, dimZ);
    
    if (dimZ == 1) {
    /* running 2D diffusion iterations */
    for(i=0; i < iterationsNumb; i++) {
            if (sigmaPar == 0.0f) LinearDiff2D(Input, Output, lambdaPar, tau, dimX, dimY); /* linear diffusion (heat equation) */
            else NonLinearDiff2D(Input, Output, lambdaPar, sigmaPar2, tau, penaltytype, dimX, dimY); /* nonlinear diffusion */
		}
	}
	else {
	/* running 3D diffusion iterations */
    for(i=0; i < iterationsNumb; i++) {
            if (sigmaPar == 0.0f) LinearDiff3D(Input, Output, lambdaPar, tau, dimX, dimY, dimZ);
            else NonLinearDiff3D(Input, Output, lambdaPar, sigmaPar2, tau, penaltytype, dimX, dimY, dimZ);
		}
	}
    return *Output;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/* linear diffusion (heat equation) */
float LinearDiff2D(float *Input, float *Output, float lambdaPar, float tau, int dimX, int dimY)
{
	int i,j,i1,i2,j1,j2,index;
	float e,w,n,s,e1,w1,n1,s1;
	
#pragma omp parallel for shared(Input) private(index,i,j,i1,i2,j1,j2,e,w,n,s,e1,w1,n1,s1)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            index = j*dimX+i;
            
                e = Output[j*dimX+i1];
                w = Output[j*dimX+i2];
                n = Output[j1*dimX+i];
                s = Output[j2*dimX+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                
                Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1) - (Output[index] - Input[index]));  
		}}
	return *Output;
}

/* nonlinear diffusion */
float NonLinearDiff2D(float *Input, float *Output, float lambdaPar, float sigmaPar, float tau, int penaltytype, int dimX, int dimY)
{
	int i,j,i1,i2,j1,j2,index;
	float e,w,n,s,e1,w1,n1,s1;
	
#pragma omp parallel for shared(Input) private(index,i,j,i1,i2,j1,j2,e,w,n,s,e1,w1,n1,s1)
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            index = j*dimX+i;
            
                e = Output[j*dimX+i1];
                w = Output[j*dimX+i2];
                n = Output[j1*dimX+i];
                s = Output[j2*dimX+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                
            if (penaltytype == 1){
            /* Huber penalty */
            if (fabs(e1) > sigmaPar) e1 =  sign(e1);
            else e1 = e1/sigmaPar;
            
            if (fabs(w1) > sigmaPar) w1 =  sign(w1);
            else w1 = w1/sigmaPar;
            
            if (fabs(n1) > sigmaPar) n1 =  sign(n1);
            else n1 = n1/sigmaPar;
            
            if (fabs(s1) > sigmaPar) s1 =  sign(s1);
            else s1 = s1/sigmaPar;
            }
            else if (penaltytype == 2) {
            /* Perona-Malik */
            e1 = (e1)/(1.0f + powf((e1/sigmaPar),2));
            w1 = (w1)/(1.0f + powf((w1/sigmaPar),2));
            n1 = (n1)/(1.0f + powf((n1/sigmaPar),2));
            s1 = (s1)/(1.0f + powf((s1/sigmaPar),2));
            }
            else if (penaltytype == 3) {
            /* Tukey Biweight */
            if (fabs(e1) <= sigmaPar) e1 =  e1*powf((1.0f - powf((e1/sigmaPar),2)), 2);
            else e1 = 0.0f;
            if (fabs(w1) <= sigmaPar) w1 =  w1*powf((1.0f - powf((w1/sigmaPar),2)), 2);
            else w1 = 0.0f;
            if (fabs(n1) <= sigmaPar) n1 =  n1*powf((1.0f - powf((n1/sigmaPar),2)), 2);
            else n1 = 0.0f;
            if (fabs(s1) <= sigmaPar) s1 =  s1*powf((1.0f - powf((s1/sigmaPar),2)), 2);
            else s1 = 0.0f;
            }
            else {
				printf("%s \n", "No penalty function selected! Use 1,2 or 3.");
				break;
				}
           Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1) - (Output[index] - Input[index]));  
		}}
	return *Output;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/* linear diffusion (heat equation) */
float LinearDiff3D(float *Input, float *Output, float lambdaPar, float tau, int dimX, int dimY, int dimZ)
{
	int i,j,k,i1,i2,j1,j2,k1,k2,index;
	float e,w,n,s,u,d,e1,w1,n1,s1,u1,d1;
	
#pragma omp parallel for shared(Input) private(index,i,j,i1,i2,j1,j2,e,w,n,s,e1,w1,n1,s1,k,k1,k2,u1,d1,u,d)
for(k=0; k<dimZ; k++) {
	k1 = k+1; if (k1 == dimZ) k1 = k-1;
    k2 = k-1; if (k2 < 0) k2 = k+1;
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            index = (dimX*dimY)*k + j*dimX+i;
            
                e = Output[(dimX*dimY)*k + j*dimX+i1];
                w = Output[(dimX*dimY)*k + j*dimX+i2];
                n = Output[(dimX*dimY)*k + j1*dimX+i];
                s = Output[(dimX*dimY)*k + j2*dimX+i];
                u = Output[(dimX*dimY)*k1 + j*dimX+i];
                d = Output[(dimX*dimY)*k2 + j*dimX+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                u1 = u - Output[index];
                d1 = d - Output[index];
                
                Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1 + u1 + d1) - (Output[index] - Input[index]));  
		}}}
	return *Output;
}

float NonLinearDiff3D(float *Input, float *Output, float lambdaPar, float sigmaPar, float tau, int penaltytype, int dimX, int dimY, int dimZ)
{
	int i,j,k,i1,i2,j1,j2,k1,k2,index;
	float e,w,n,s,u,d,e1,w1,n1,s1,u1,d1;
	
#pragma omp parallel for shared(Input) private(index,i,j,i1,i2,j1,j2,e,w,n,s,e1,w1,n1,s1,k,k1,k2,u1,d1,u,d)
for(k=0; k<dimZ; k++) {
	k1 = k+1; if (k1 == dimZ) k1 = k-1;
    k2 = k-1; if (k2 < 0) k2 = k+1;
    for(i=0; i<dimX; i++) {
        /* symmetric boundary conditions (Neuman) */
        i1 = i+1; if (i1 == dimX) i1 = i-1;
        i2 = i-1; if (i2 < 0) i2 = i+1;
        for(j=0; j<dimY; j++) {
            /* symmetric boundary conditions (Neuman) */
            j1 = j+1; if (j1 == dimY) j1 = j-1;
            j2 = j-1; if (j2 < 0) j2 = j+1;
            index = (dimX*dimY)*k + j*dimX+i;
            
                e = Output[(dimX*dimY)*k + j*dimX+i1];
                w = Output[(dimX*dimY)*k + j*dimX+i2];
                n = Output[(dimX*dimY)*k + j1*dimX+i];
                s = Output[(dimX*dimY)*k + j2*dimX+i];
                u = Output[(dimX*dimY)*k1 + j*dimX+i];
                d = Output[(dimX*dimY)*k2 + j*dimX+i];
                
                e1 = e - Output[index];
                w1 = w - Output[index];
                n1 = n - Output[index];
                s1 = s - Output[index];
                u1 = u - Output[index];
                d1 = d - Output[index];
                
             if (penaltytype == 1){
            /* Huber penalty */
            if (fabs(e1) > sigmaPar) e1 =  sign(e1);
            else e1 = e1/sigmaPar;
            
            if (fabs(w1) > sigmaPar) w1 =  sign(w1);
            else w1 = w1/sigmaPar;
            
            if (fabs(n1) > sigmaPar) n1 =  sign(n1);
            else n1 = n1/sigmaPar;
            
            if (fabs(s1) > sigmaPar) s1 =  sign(s1);
            else s1 = s1/sigmaPar;
            
            if (fabs(u1) > sigmaPar) u1 =  sign(u1);
            else u1 = u1/sigmaPar;
            
            if (fabs(d1) > sigmaPar) d1 =  sign(d1);
            else d1 = d1/sigmaPar;            
            }
            else if (penaltytype == 2) {
            /* Perona-Malik */
            e1 = (e1)/(1.0f + powf((e1/sigmaPar),2));
            w1 = (w1)/(1.0f + powf((w1/sigmaPar),2));
            n1 = (n1)/(1.0f + powf((n1/sigmaPar),2));
            s1 = (s1)/(1.0f + powf((s1/sigmaPar),2));
            u1 = (u1)/(1.0f + powf((u1/sigmaPar),2));
            d1 = (d1)/(1.0f + powf((d1/sigmaPar),2));
            }
            else if (penaltytype == 3) {
            /* Tukey Biweight */
            if (fabs(e1) <= sigmaPar) e1 =  e1*powf((1.0f - powf((e1/sigmaPar),2)), 2);
            else e1 = 0.0f;
            if (fabs(w1) <= sigmaPar) w1 =  w1*powf((1.0f - powf((w1/sigmaPar),2)), 2);
            else w1 = 0.0f;
            if (fabs(n1) <= sigmaPar) n1 =  n1*powf((1.0f - powf((n1/sigmaPar),2)), 2);
            else n1 = 0.0f;
            if (fabs(s1) <= sigmaPar) s1 =  s1*powf((1.0f - powf((s1/sigmaPar),2)), 2);
            else s1 = 0.0f;
            if (fabs(u1) <= sigmaPar) u1 =  u1*powf((1.0f - powf((u1/sigmaPar),2)), 2);
            else u1 = 0.0f;
            if (fabs(d1) <= sigmaPar) d1 =  d1*powf((1.0f - powf((d1/sigmaPar),2)), 2);
            else d1 = 0.0f;
            }
            else {
				printf("%s \n", "No penalty function selected! Use 1,2 or 3.");
				break;
				}

                Output[index] += tau*(lambdaPar*(e1 + w1 + n1 + s1 + u1 + d1) - (Output[index] - Input[index]));  
		}}}
	return *Output;
}

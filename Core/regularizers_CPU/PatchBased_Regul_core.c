/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazanteev
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

#include "PatchBased_Regul_core.h"

/* C-OMP implementation of  patch-based (PB) regularization (2D and 3D cases). 
 * This method finds self-similar patches in data and performs one fixed point iteration to mimimize the PB penalty function
 * 
 * References: 1. Yang Z. & Jacob M. "Nonlocal Regularization of Inverse Problems"
 *             2. Kazantsev D. et al. "4D-CT reconstruction with unified spatial-temporal patch-based regularization"
 *
 * Input Parameters:
 * 1. Image (2D or 3D) [required]
 * 2. ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window) [optional]
 * 3. ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window) [optional]
 * 4. h - parameter for the PB penalty function [optional]
 * 5. lambda - regularization parameter  [optional]

 * Output:
 * 1. regularized (denoised) Image (N x N)/volume (N x N x N)
 *
 * 2D denoising example in Matlab:   
   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
   ImDen = PatchBased_Regul(single(u0), 3, 1, 0.08, 0.05); 
 
 * D. Kazantsev *
 * 02/07/2014
 * Harwell, UK
 */

/*2D version function */
float PB_FUNC2D(float *A, float *B, int dimX, int dimY, int padXY, int SearchW, int SimilW, float h, float lambda)
{
    int i, j, i_n, j_n, i_m, j_m, i_p, j_p, i_l, j_l, i1, j1, i2, j2, i3, j3, i5,j5, count, SimilW_full;
    float *Eucl_Vec, h2, denh2, normsum, Weight, Weight_norm, value, denom, WeightGlob, t1;
                 
    /*SearchW_full = 2*SearchW + 1; */ /* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   /* the full similarity window  size */
    h2 = h*h;
    denh2 = 1/(2*h2);   
    
     /*Gaussian kernel */
    Eucl_Vec = (float*) calloc (SimilW_full*SimilW_full,sizeof(float));
    count = 0;
    for(i_n=-SimilW; i_n<=SimilW; i_n++) {
        for(j_n=-SimilW; j_n<=SimilW; j_n++) {
            t1 = pow(((float)i_n), 2) + pow(((float)j_n), 2);
            Eucl_Vec[count] = exp(-(t1)/(2*SimilW*SimilW));
            count = count + 1;                       
        }} /*main neighb loop */   
    
    /*The NLM code starts here*/         
    /* setting OMP here */
    #pragma omp parallel for shared (A, B, dimX, dimY, Eucl_Vec, lambda, denh2) private(denom, i, j, WeightGlob, count,  i1, j1, i2, j2, i3, j3, i5, j5, Weight_norm, normsum, i_m, j_m, i_n, j_n, i_l, j_l, i_p, j_p, Weight,  value)
    
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
             if (((i >= padXY) && (i < dimX-padXY)) &&  ((j >= padXY) && (j < dimY-padXY))) {
          
                /* Massive Search window loop */
                Weight_norm = 0; value = 0.0;
                for(i_m=-SearchW; i_m<=SearchW; i_m++) {
                    for(j_m=-SearchW; j_m<=SearchW; j_m++) {
                       /*checking boundaries*/
                        i1 = i+i_m; j1 = j+j_m;
                        
                        WeightGlob = 0.0;
                        /* if inside the searching window */                        
                         for(i_l=-SimilW; i_l<=SimilW; i_l++) {
                             for(j_l=-SimilW; j_l<=SimilW; j_l++) {
                                 i2 = i1+i_l; j2 = j1+j_l;
                                 
                                 i3 = i+i_l; j3 = j+j_l;   /*coordinates of the inner patch loop */
                                
                                 count = 0; normsum = 0.0;
                                 for(i_p=-SimilW; i_p<=SimilW; i_p++) {
                                     for(j_p=-SimilW; j_p<=SimilW; j_p++) {
                                         i5 = i2 + i_p; j5 = j2 + j_p;
                                         normsum = normsum + Eucl_Vec[count]*pow(A[(i3+i_p)*dimY+(j3+j_p)]-A[i5*dimY+j5], 2);        
                                         count = count + 1;
                                     }}
                                  if (normsum != 0) Weight = (exp(-normsum*denh2)); 
                                  else Weight = 0.0;
                                 WeightGlob += Weight;
                             }}                      
      
                         value += A[i1*dimY+j1]*WeightGlob;
                         Weight_norm += WeightGlob;           
                    }}      /*search window loop end*/
                
                /* the final loop to average all values in searching window with weights */
                denom = 1 + lambda*Weight_norm;
                B[i*dimY+j] = (A[i*dimY+j] + lambda*value)/denom;         
             }
        }}   /*main loop*/     
    return (*B);
    free(Eucl_Vec);    
}
 
/*3D version*/ 
 float PB_FUNC3D(float *A, float *B, int dimX, int dimY, int dimZ, int padXY, int SearchW, int SimilW, float h, float lambda)       
 {
    int SimilW_full, count, i, j, k,  i_n, j_n, k_n, i_m, j_m, k_m, i_p, j_p, k_p, i_l, j_l, k_l, i1, j1, k1, i2, j2, k2, i3, j3, k3, i5, j5, k5;
    float *Eucl_Vec, h2, denh2, normsum, Weight, Weight_norm, value, denom, WeightGlob;
        
    /*SearchW_full = 2*SearchW + 1; */ /* the full searching window  size */
    SimilW_full = 2*SimilW + 1;   /* the full similarity window  size */
    h2 = h*h;
    denh2 = 1/(2*h2);
    
    /*Gaussian kernel */
    Eucl_Vec = (float*) calloc (SimilW_full*SimilW_full*SimilW_full,sizeof(float));
    count = 0;
    for(i_n=-SimilW; i_n<=SimilW; i_n++) {
        for(j_n=-SimilW; j_n<=SimilW; j_n++) {
            for(k_n=-SimilW; k_n<=SimilW; k_n++) {
                Eucl_Vec[count] = exp(-(pow((float)i_n, 2) + pow((float)j_n, 2) + pow((float)k_n, 2))/(2*SimilW*SimilW*SimilW));
                count = count + 1;
            }}} /*main neighb loop */
    
    /*The NLM code starts here*/         
    /* setting OMP here */
    #pragma omp parallel for shared (A, B, dimX, dimY, dimZ, Eucl_Vec, lambda, denh2) private(denom, i, j, k, WeightGlob,count,  i1, j1, k1, i2, j2, k2, i3, j3, k3, i5, j5, k5, Weight_norm, normsum, i_m, j_m, k_m, i_n, j_n, k_n, i_l, j_l, k_l, i_p, j_p, k_p, Weight, value)    
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            for(k=0; k<dimZ; k++) {
            if (((i >= padXY) && (i < dimX-padXY)) &&  ((j >= padXY) && (j < dimY-padXY)) &&  ((k >= padXY) && (k < dimZ-padXY))) {
            /* take all elements around the pixel of interest */                             
               /* Massive Search window loop */
                Weight_norm = 0;  value = 0.0;
                for(i_m=-SearchW; i_m<=SearchW; i_m++) {
                    for(j_m=-SearchW; j_m<=SearchW; j_m++) {
                        for(k_m=-SearchW; k_m<=SearchW; k_m++) {
                         /*checking boundaries*/
                        i1 = i+i_m; j1 = j+j_m; k1 = k+k_m;
                        
                        WeightGlob = 0.0;
                        /* if inside the searching window */                        
                         for(i_l=-SimilW; i_l<=SimilW; i_l++) {
                             for(j_l=-SimilW; j_l<=SimilW; j_l++) {
                                 for(k_l=-SimilW; k_l<=SimilW; k_l++) {                                 
                                 i2 = i1+i_l; j2 = j1+j_l; k2 = k1+k_l;
                                 
                                 i3 = i+i_l; j3 = j+j_l; k3 = k+k_l;   /*coordinates of the inner patch loop */
                                
                                 count = 0; normsum = 0.0;
                                 for(i_p=-SimilW; i_p<=SimilW; i_p++) {
                                     for(j_p=-SimilW; j_p<=SimilW; j_p++) {
                                         for(k_p=-SimilW; k_p<=SimilW; k_p++) {
                                         i5 = i2 + i_p; j5 = j2 + j_p; k5 = k2 + k_p;
                                         normsum = normsum + Eucl_Vec[count]*pow(A[(dimX*dimY)*(k3+k_p)+(i3+i_p)*dimY+(j3+j_p)]-A[(dimX*dimY)*k5 + i5*dimY+j5], 2);        
                                         count = count + 1;
                                     }}}
                                  if (normsum != 0) Weight = (exp(-normsum*denh2)); 
                                  else Weight = 0.0;
                                 WeightGlob += Weight;
                             }}}                                                 
                         value += A[(dimX*dimY)*k1 + i1*dimY+j1]*WeightGlob;
                         Weight_norm += WeightGlob;
             
                    }}}      /*search window loop end*/
                
                /* the final loop to average all values in searching window with weights */
                denom = 1 + lambda*Weight_norm;               
                B[(dimX*dimY)*k + i*dimY+j] = (A[(dimX*dimY)*k + i*dimY+j] + lambda*value)/denom;      
            }            
        }}}   /*main loop*/              
       free(Eucl_Vec);        
       return *B;
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

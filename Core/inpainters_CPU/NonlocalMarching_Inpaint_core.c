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

#include "NonlocalMarching_Inpaint_core.h"
#include "utils.h"


/* C-OMP implementation of Nonlocal Vertical Marching inpainting method (2D case)
 * The method is heuristic but computationally efficent (especially for larger images).
 * It developed specifically to smoothly inpaint horizontal or inclined missing data regions in sinograms
 * The method WILL not work satisfactory if you have lengthy vertical stripes of missing data
 *
 * Input:
 * 1. 2D image or sinogram with horizontal or inclined regions of missing data
 * 2. Mask of the same size as A in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Linear increment to increase searching window size in iterations, values from 1-3 is a good choice
 
 * Output:
 * 1. Inpainted image or a sinogram
 * 2. updated mask
 *
 * Reference: TBA
 */

float NonlocalMarching_Inpaint_main(float *Input, unsigned char *M, float *Output, unsigned char *M_upd, int SW_increment, int iterationsNumb, int dimX, int dimY, int dimZ)
{
    int i, j, i_m, j_m, counter, iter, iterations_number, W_fullsize, switchmask, switchcurr, counterElements;
    float *Gauss_weights;
    
    /* copying M to M_upd */
    copyIm_unchar(M, M_upd, dimX, dimY, 1);
    
    /* Copying the image */
    copyIm(Input, Output, dimX, dimY, 1);
    
    /* Find how many inpainting iterations (equal to the number of ones) required based on a mask  */
    if (iterationsNumb == 0) {
        iterations_number = 0;
        for (i=0; i<dimY*dimX; i++) {
            if (M[i] == 1) iterations_number++;
        }
    }
    else iterations_number = (int)(iterationsNumb/dimX);
    if (iterations_number > dimX) iterations_number = dimX;
    
    if (iterations_number == 0) printf("%s \n", "Nothing to inpaint, zero mask!");
    else {
        
        printf("%s %i \n", "Max iteration number equals to:", iterations_number);
        
        /* Inpainting iterations run here*/
        int W_halfsize = 1;
        for(iter=0; iter < iterations_number; iter++) {
            
            //if (mod (iter, 2) == 0) {W_halfsize += 1;}
            // printf("%i \n", W_halfsize);
            
            /* pre-calculation of Gaussian distance weights  */
            W_fullsize = (int)(2*W_halfsize + 1); /*full size of similarity window */
            Gauss_weights = (float*)calloc(W_fullsize*W_fullsize,sizeof(float ));
            counter = 0;
            for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
                for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
                    Gauss_weights[counter] = exp(-(pow((i_m), 2) + pow((j_m), 2))/(2*W_halfsize*W_halfsize));
                    counter++;
                }
            }
            /* find a point in the mask to inpaint */
#pragma omp parallel for shared(Output, M_upd, Gauss_weights) private(i, j, switchmask, switchcurr)            
            for(j=0; j<dimY; j++) {
                switchmask = 0;
                for(i=0; i<dimX; i++) {                
                    switchcurr = 0;
                    if ((M_upd[j*dimX + i] == 1) && (switchmask == 0)) {
                        /* perform inpainting of the current pixel */
                        inpaint_func(Output, M_upd, Gauss_weights, i, j, dimX, dimY, W_halfsize, W_fullsize);
                        /* add value to the mask*/
                        M_upd[j*dimX + i] = 0;
                        switchmask = 1; switchcurr = 1;
                    }
                    if ((M_upd[j*dimX + i] == 0) && (switchmask == 1) && (switchcurr == 0)) {                        
                        /* perform inpainting of the previous (j-1) pixel */
                        inpaint_func(Output, M_upd, Gauss_weights, i-1, j, dimX, dimY, W_halfsize, W_fullsize);
                        /* add value to the mask*/
                        M_upd[(j)*dimX + i-1] = 0;                 
                        switchmask = 0;                        
                    }
                }
            }
            free(Gauss_weights);
            
            /* check if possible to terminate iterations earlier */
            counterElements = 0;
            for(i=0; i<dimX*dimY; i++) if (M_upd[i] == 0) counterElements++;
            
            if (counterElements == dimX*dimY) {
                printf("%s \n", "Padding completed!");
                break;
            }
            W_halfsize += SW_increment;
        }
        printf("%s %i \n", "Iterations stopped at:", iter);
    }
    return *Output;
}

float inpaint_func(float *U, unsigned char *M_upd, float *Gauss_weights, int i, int j, int dimX, int dimY, int W_halfsize, int W_fullsize)
{
    int i1, j1, i_m, j_m, counter;
    float sum_val, sumweight;
    
    /*method 1: inpainting based on Euclidian weights */
    sumweight = 0.0f;
    counter = 0; sum_val = 0.0f;
    for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
        i1 = i+i_m;
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                if (M_upd[j1*dimX + i1] == 0) {
                    sumweight += Gauss_weights[counter];
                }
            }
            counter++;
        }
    }
    counter = 0; sum_val = 0.0f;
    for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
        i1 = i+i_m;
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                if ((M_upd[j1*dimX + i1] == 0) && (sumweight != 0.0f)) {
                    /* we have data so add it with Euc weight */
                    sum_val += (Gauss_weights[counter]/sumweight)*U[j1*dimX + i1];
                }
            }
            counter++;
        }
    }
    U[j*dimX + i] = sum_val;
    return *U;
}


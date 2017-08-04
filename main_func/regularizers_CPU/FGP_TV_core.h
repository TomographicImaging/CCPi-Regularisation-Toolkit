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

//#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
*
* Input Parameters:
* 1. Noisy image/volume [REQUIRED]
* 2. lambda - regularization parameter [REQUIRED]
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon: tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
*
* Output:
* [1] Filtered/regularized image
* [2] last function value
*
* Example of image denoising:
* figure;
* Im = double(imread('lena_gray_256.tif'))/255;  % loading image
* u0 = Im + .05*randn(size(Im)); % adding noise
* u = FGP_TV(single(u0), 0.05, 100, 1e-04);
*
* to compile with OMP support: mex FGP_TV.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
* This function is based on the Matlab's code and paper by
* [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
*
* D. Kazantsev, 2016-17
*
*/
#ifdef __cplusplus
extern "C" {
#endif
//float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);
float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
float Proj_func2D(float *P1, float *P2, int methTV, int dimX, int dimY);
float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int dimX, int dimY);

float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ);
float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ);
float Proj_func3D(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ);
float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, int dimX, int dimY, int dimZ);
#ifdef __cplusplus
}
#endif
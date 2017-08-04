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

/* C-OMP implementation of Primal-Dual denoising method for
* Total Generilized Variation (TGV)-L2 model (2D case only)
*
* Input Parameters:
* 1. Noisy image/volume (2D)
* 2. lambda - regularization parameter
* 3. parameter to control first-order term (alpha1)
* 4. parameter to control the second-order term (alpha0)
* 5. Number of CP iterations
*
* Output:
* Filtered/regularized image
*
* Example:
* figure;
* Im = double(imread('lena_gray_256.tif'))/255;  % loading image
* u0 = Im + .03*randn(size(Im)); % adding noise
* tic; u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550); toc;
*
* to compile with OMP support: mex TGV_PD.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
* References:
* K. Bredies "Total Generalized Variation"
*
* 28.11.16/Harwell
*/
#ifdef __cplusplus
extern "C" {
#endif
/* 2D functions */
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, int dimZ, float sigma);
float ProjP_2D(float *P1, float *P2, int dimX, int dimY, int dimZ, float alpha1);
float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float sigma);
float ProjQ_2D(float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float alpha0);
float DivProjP_2D(float *U, float *A, float *P1, float *P2, int dimX, int dimY, int dimZ, float lambda, float tau);
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float tau);
float newU(float *U, float *U_old, int dimX, int dimY, int dimZ);
//float copyIm(float *A, float *U, int dimX, int dimY, int dimZ);
#ifdef __cplusplus
}
#endif

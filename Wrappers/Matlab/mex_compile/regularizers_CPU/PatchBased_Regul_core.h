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

#define _USE_MATH_DEFINES

//#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

/* C-OMP implementation of  patch-based (PB) regularization (2D and 3D cases).
* This method finds self-similar patches in data and performs one fixed point iteration to mimimize the PB penalty function
*
* References: 1. Yang Z. & Jacob M. "Nonlocal Regularization of Inverse Problems"
*             2. Kazantsev D. et al. "4D-CT reconstruction with unified spatial-temporal patch-based regularization"
*
* Input Parameters (mandatory):
* 1. Image (2D or 3D)
* 2. ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window)
* 3. ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window)
* 4. h - parameter for the PB penalty function
* 5. lambda - regularization parameter

* Output:
* 1. regularized (denoised) Image (N x N)/volume (N x N x N)
*
* Quick 2D denoising example in Matlab:
Im = double(imread('lena_gray_256.tif'))/255;  % loading image
u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
ImDen = PB_Regul_CPU(single(u0), 3, 1, 0.08, 0.05);
*
* Please see more tests in a file:
TestTemporalSmoothing.m

*
* Matlab + C/mex compilers needed
* to compile with OMP support: mex PB_Regul_CPU.c CFLAGS="\$CFLAGS -fopenmp -Wall" LDFLAGS="\$LDFLAGS -fopenmp"
*
* D. Kazantsev *
* 02/07/2014
* Harwell, UK
*/
#ifdef __cplusplus
extern "C" {
#endif
float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY, int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop);
float PB_FUNC2D(float *A, float *B, int dimX, int dimY, int padXY, int SearchW, int SimilW, float h, float lambda);
float PB_FUNC3D(float *A, float *B, int dimX, int dimY, int dimZ, int padXY, int SearchW, int SimilW, float h, float lambda);
#ifdef __cplusplus
}
#endif
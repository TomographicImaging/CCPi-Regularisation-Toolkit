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
#include "CCPiDefines.h"

/* C-OMP implementation of FGP-TV [1] denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default)
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]

 *
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 */

int apply_nonnegativity(float *A, long DimTotal);
int calculate_norm(const float * A, const float * A_prev, float * re, long DimTotal);
int Grad_func(float *P1, float *P2, const float *D, const float *R1, const float *R2, float lambda, long dimX, long dimY);
int Grad_func(float *P1, float *P2, float *P3, const float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ);
int Rupd_func(const float *P1, const float *P1_old, const float *P2, const float *P2_old, float *R1, float *R2, float tkp1, float tk, long DimTotal);
int Rupd_func(const float *P1, const float *P1_old, const float *P2, const float *P2_old, const float *P3, const float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, long DimTotal);
int Obj_func(const float *A, float *D, const float *R1, const float *R2, float lambda, long dimX, long dimY);
int Obj_func(const float *A, float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ);

#ifdef __cplusplus
extern "C" {
#endif

CCPI_EXPORT float TV_FGP_CPU(const float *Input, float *Output, float *infovector, float lambdaPar, int iterationsNumb, float epsil, int methodTV, int nonneg, int dimX, int dimY, int dimZ);

#ifdef __cplusplus
}
#endif

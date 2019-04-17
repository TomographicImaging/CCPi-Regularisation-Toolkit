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

#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"
#include "CCPiDefines.h"

/* C-OMP implementation of ROF-TV denoising/regularization model [1] (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. lambda - regularization parameter (a constant or the same size as input (1))
 * 3. tau - marching step for explicit scheme, ~1 is recommended [REQUIRED]
 * 4. Number of iterations, for explicit scheme >= 150 is recommended  [REQUIRED]
 * 5. eplsilon: tolerance constant
 *
 * Output:
 * [1] Regularised image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
 */

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float TV_ROF_CPU_main(float *Input, float *Output, float *infovector, float *lambdaPar, int lambda_is_arr, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
CCPI_EXPORT float TV_kernel(float *D1, float *D2, float *D3, float *B, float *A, float *lambda, int lambda_is_arr, float tau, long dimX, long dimY, long dimZ);
CCPI_EXPORT float D1_func(float *A, float *D1, long dimX, long dimY, long dimZ);
CCPI_EXPORT float D2_func(float *A, float *D2, long dimX, long dimY, long dimZ);
CCPI_EXPORT float D3_func(float *A, float *D3, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif

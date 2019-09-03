/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC and Diamond Light Source Ltd.
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 * Copyright 2018 Diamond Light Source Ltd.
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

#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"
#include "CCPiDefines.h"

#define EPS 1.0000e-9

/* C-OMP implementation of non-local regulariser
 * Weights and associated indices must be given as an input.
 * Gauss-Seidel fixed point iteration requires ~ 3 iterations, so the main effort
 * goes in pre-calculation of weights and selection of patches
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. AR_i - indeces of i neighbours
 * 3. AR_j - indeces of j neighbours
 * 4. AR_k - indeces of k neighbours (0 - for 2D case)
 * 5. Weights_ij(k) - associated weights
 * 6. regularisation parameter
 * 7. iterations number

 * Output:
 * 1. denoised image/volume
 * Elmoataz, Abderrahim, Olivier Lezoray, and Sébastien Bougleux. "Nonlocal discrete regularization on weighted graphs: a framework for image and manifold processing." IEEE Trans.   Image Processing 17, no. 7 (2008): 1047-1060.
 */

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float Nonlocal_TV_CPU_main(float *A_orig, float *Output, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, int dimX, int dimY, int dimZ, int NumNeighb, float lambdaReg, int IterNumb, int switchM);
CCPI_EXPORT float NLM_H1_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, int NumNeighb, float lambdaReg);
CCPI_EXPORT float NLM_TV_2D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, int NumNeighb, float lambdaReg);
CCPI_EXPORT float NLM_H1_3D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimX, long dimY, long dimZ, int NumNeighb, float lambdaReg);
CCPI_EXPORT float NLM_TV_3D(float *A, float *A_orig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimX, long dimY, long dimZ, int NumNeighb, float lambdaReg);
#ifdef __cplusplus
}
#endif

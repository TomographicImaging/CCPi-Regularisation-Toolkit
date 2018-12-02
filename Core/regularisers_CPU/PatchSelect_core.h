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
#define EPS 1.0000e-12

/* C-OMP implementation of non-local weight pre-calculation for non-local priors
 * Weights and associated indices are stored into pre-allocated arrays and passed
 * to the regulariser
 *
 *
 * Input Parameters:
 * 1. 2D/3D grayscale image/volume
 * 2. Searching window (half-size of the main bigger searching window, e.g. 11)
 * 3. Similarity window (half-size of the patch window, e.g. 2)
 * 4. The number of neighbours to take (the most prominent after sorting neighbours will be taken)
 * 5. noise-related parameter to calculate non-local weights
 *
 * Output [2D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. Weights_ij - associated weights
 *
 * Output [3D]:
 * 1. AR_i - indeces of i neighbours
 * 2. AR_j - indeces of j neighbours
 * 3. AR_k - indeces of j neighbours
 * 4. Weights_ijk - associated weights
 */
/*****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float PatchSelect_CPU_main(float *A, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, int dimX, int dimY, int dimZ, int SearchWindow, int SimilarWin, int NumNeighb, float h, int switchM);
CCPI_EXPORT float Indeces2D(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2);
CCPI_EXPORT float Indeces2D_p(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, long i, long j, long dimX, long dimY, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2);
CCPI_EXPORT float Indeces3D(float *Aorig, unsigned short *H_i, unsigned short *H_j, unsigned short *H_k, float *Weights, long i, long j, long k, long dimY, long dimX, long dimZ, float *Eucl_Vec, int NumNeighb, int SearchWindow, int SimilarWin, float h2);
#ifdef __cplusplus
}
#endif

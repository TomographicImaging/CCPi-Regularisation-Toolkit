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

/* C-OMP implementation of Lysaker, Lundervold and Tai (LLT) model [1] combined with Rudin-Osher-Fatemi [2] TV regularisation penalty.
 * 
* This penalty can deliver visually pleasant piecewise-smooth recovery if regularisation parameters are selected well. 
* The rule of thumb for selection is to start with lambdaLLT = 0 (just the ROF-TV model) and then proceed to increase 
* lambdaLLT starting with smaller values. 
*
* Input Parameters:
* 1. U0 - original noise image/volume
* 2. lambdaROF - ROF-related regularisation parameter
* 3. lambdaLLT - LLT-related regularisation parameter
* 4. tau - time-marching step 
* 5. iter - iterations number (for both models)
*
* Output:
* Filtered/regularised image
*
* References: 
* [1] Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.
* [2] Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms"
*/

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float LLT_ROF_CPU_main(float *Input, float *Output, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau, int dimX, int dimY, int dimZ);

CCPI_EXPORT float der2D_LLT(float *U, float *D1, float *D2, int dimX, int dimY, int dimZ);
CCPI_EXPORT float der3D_LLT(float *U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ);

CCPI_EXPORT float D1_func_ROF(float *A, float *D1, int dimY, int dimX, int dimZ);
CCPI_EXPORT float D2_func_ROF(float *A, float *D2, int dimY, int dimX, int dimZ);
CCPI_EXPORT float D3_func_ROF(float *A, float *D3, int dimY, int dimX, int dimZ);

CCPI_EXPORT float Update2D_LLT_ROF(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D1_ROF, float *D2_ROF, float lambdaROF, float lambdaLLT, float tau, int dimX, int dimY, int dimZ);
CCPI_EXPORT float Update3D_LLT_ROF(float *U0, float *U, float *D1_LLT, float *D2_LLT, float *D3_LLT, float *D1_ROF, float *D2_ROF, float *D3_ROF, float lambdaROF, float lambdaLLT, float tau, int dimX, int dimY, int dimZ);
#ifdef __cplusplus
}
#endif

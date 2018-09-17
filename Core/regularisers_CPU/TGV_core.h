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

/* C-OMP implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model [1] (2D case only)
 *
 * Input Parameters:
 * 1. Noisy image (2D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 * 
 * Output:
 * Filtered/regulariaed image 
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */
 
 
#ifdef __cplusplus
extern "C" {
#endif
/* 2D functions */
CCPI_EXPORT float TGV_main(float *U0, float *U, float lambda, float alpha1, float alpha0, int iter, float L2, int dimX, int dimY);

CCPI_EXPORT float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, long dimX, long dimY, float sigma);
CCPI_EXPORT float ProjP_2D(float *P1, float *P2, long dimX, long dimY, float alpha1);
CCPI_EXPORT float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float sigma);
CCPI_EXPORT float ProjQ_2D(float *Q1, float *Q2, float *Q3, long dimX, long dimY, float alpha0);
CCPI_EXPORT float DivProjP_2D(float *U, float *U0, float *P1, float *P2, long dimX, long dimY, float lambda, float tau);
CCPI_EXPORT float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, long dimX, long dimY, float tau);
CCPI_EXPORT float newU(float *U, float *U_old, long dimX, long dimY);
#ifdef __cplusplus
}
#endif

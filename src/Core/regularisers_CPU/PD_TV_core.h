/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev
Copyright 2019 Srikanth Nagella, Edoardo Pasca

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

/* C-OMP implementation of Primal-Dual TV [1] by Chambolle Pock denoising/regularization model (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. lipschitz_const: convergence related parameter
 * 6. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 7. nonneg: 'nonnegativity (0 is OFF by default, 1 is ON)

 * Output:
 * [1] TV - Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * [1] Antonin Chambolle, Thomas Pock. "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", 2010
 */

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float PDTV_CPU_main(float *Input, float *U, float *infovector, float lambdaPar, int iterationsNumb, float epsil, float lipschitz_const, int methodTV, int nonneg, int dimX, int dimY, int dimZ);

CCPI_EXPORT float DualP2D(float *U, float *P1, float *P2, long dimX, long dimY, float sigma);
CCPI_EXPORT float DivProj2D(float *U, float *Input, float *P1, float *P2, long dimX, long dimY, float lt, float tau);
CCPI_EXPORT float getX2D(float *U, float *U_old, float theta, long DimTotal);
#ifdef __cplusplus
}
#endif

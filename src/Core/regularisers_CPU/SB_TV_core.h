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


/* C-OMP implementation of Split Bregman - TV denoising-regularisation model (2D/3D) [1]
*
* Input Parameters:
* 1. Noisy image/volume
* 2. lambda - regularisation parameter
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon - tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]

* Output:
* [1] Filtered/regularized image/volume
* [2] Information vector which contains [iteration no., reached tolerance]
*
* [1]. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.
*/

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float SB_TV_CPU_main(float *Input, float *Output, float *infovector, float mu, int iter, float epsil, int methodTV, int dimX, int dimY, int dimZ);

CCPI_EXPORT float gauss_seidel2D(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Bx, float *By, long dimX, long dimY, float lambda, float mu);
CCPI_EXPORT float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, long dimX, long dimY, float lambda);
CCPI_EXPORT float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, long dimX, long dimY, float lambda);
CCPI_EXPORT float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, long dimX, long dimY);

CCPI_EXPORT float gauss_seidel3D(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, long dimX, long dimY, long dimZ, float lambda, float mu);
CCPI_EXPORT float updDxDyDz_shrinkAniso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, long dimX, long dimY, long dimZ, float lambda);
CCPI_EXPORT float updDxDyDz_shrinkIso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, long dimX, long dimY, long dimZ, float lambda);
CCPI_EXPORT float updBxByBz3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif

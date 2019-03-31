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

/* C-OMP implementation of fourth-order diffusion scheme [1] for piecewise-smooth recovery (2D/3D case)
 * The minimisation is performed using explicit scheme.
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambda - regularization parameter
 * 3. Edge-preserving parameter (sigma)
 * 4. Number of iterations, for explicit scheme >= 150 is recommended
 * 5. tau - time-marching step for explicit scheme
 * 6. eplsilon: tolerance constant
 *
 * Output:
 * [1] Regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the paper by
 * [1] Hajiaboli, M.R., 2011. An anisotropic fourth-order diffusion filter for image noise removal. International Journal of Computer Vision, 92(2), pp.177-191.
 */

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float Diffus4th_CPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, float epsil, int dimX, int dimY, int dimZ);
CCPI_EXPORT float Weighted_Laplc2D(float *W_Lapl, float *U0, float sigma, long dimX, long dimY);
CCPI_EXPORT float Diffusion_update_step2D(float *Output, float *Input, float *W_Lapl, float lambdaPar, float sigmaPar2, float tau, long dimX, long dimY);
CCPI_EXPORT float Weighted_Laplc3D(float *W_Lapl, float *U0, float sigma, long dimX, long dimY, long dimZ);
CCPI_EXPORT float Diffusion_update_step3D(float *Output, float *Input, float *W_Lapl, float lambdaPar, float sigmaPar2, float tau, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif

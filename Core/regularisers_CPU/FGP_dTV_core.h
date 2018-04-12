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

/* C-OMP implementation of FGP-dTV [1,2] denoising/regularization model (2D/3D case)
 * which employs structural similarity of the level sets of two images/volumes, see [1,2]
 * The current implementation updates image 1 while image 2 is being fixed.
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. Additional reference image/volume of the same dimensions as (1) [REQUIRED]
 * 3. lambdaPar - regularization parameter [REQUIRED]
 * 4. Number of iterations [OPTIONAL]
 * 5. eplsilon: tolerance constant [OPTIONAL]
 * 6. eta: smoothing constant to calculate gradient of the reference [OPTIONAL] * 
 * 7. TV-type: methodTV - 'iso' (0) or 'l1' (1) [OPTIONAL]
 * 8. nonneg: 'nonnegativity (0 is OFF by default) [OPTIONAL]
 * 9. print information: 0 (off) or 1 (on) [OPTIONAL]
 *
 * Output:
 * [1] Filtered/regularized image/volume
 *
 * This function is based on the Matlab's codes and papers by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 * [2] M. J. Ehrhardt and M. M. Betcke, Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation, SIAM Journal on Imaging Sciences 9(3), pp. 1084–1106
 */
 
#ifdef __cplusplus
extern "C" {
#endif
float dTV_FGP_CPU_main(float *Input, float *InputRef, float *Output, float lambdaPar, int iterationsNumb, float epsil, float eta, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);

CCPI_EXPORT float GradNorm_func2D(float *B, float *B_x, float *B_y, float eta, int dimX, int dimY);
CCPI_EXPORT float ProjectVect_func2D(float *R1, float *R2, float *B_x, float *B_y, int dimX, int dimY);
CCPI_EXPORT float Obj_func2D(float *A, float *D, float *R1, float *R2, float lambda, int dimX, int dimY);
CCPI_EXPORT float Grad_func2D(float *P1, float *P2, float *D, float *R1, float *R2, float *B_x, float *B_y, float lambda, int dimX, int dimY);
CCPI_EXPORT float Proj_func2D(float *P1, float *P2, int methTV, int DimTotal);
CCPI_EXPORT float Rupd_func2D(float *P1, float *P1_old, float *P2, float *P2_old, float *R1, float *R2, float tkp1, float tk, int DimTotal);

CCPI_EXPORT float GradNorm_func3D(float *B, float *B_x, float *B_y, float *B_z, float eta, int dimX, int dimY, int dimZ);
CCPI_EXPORT float ProjectVect_func3D(float *R1, float *R2, float *R3, float *B_x, float *B_y, float *B_z, int dimX, int dimY, int dimZ);
CCPI_EXPORT float Obj_func3D(float *A, float *D, float *R1, float *R2, float *R3, float lambda, int dimX, int dimY, int dimZ);
CCPI_EXPORT float Grad_func3D(float *P1, float *P2, float *P3, float *D, float *R1, float *R2, float *R3, float *B_x, float *B_y, float *B_z, float lambda, int dimX, int dimY, int dimZ);
CCPI_EXPORT float Proj_func3D(float *P1, float *P2, float *P3, int methTV, int DimTotal);
CCPI_EXPORT float Rupd_func3D(float *P1, float *P1_old, float *P2, float *P2_old, float *P3, float *P3_old, float *R1, float *R2, float *R3, float tkp1, float tk, int DimTotal);
#ifdef __cplusplus
}
#endif

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


/* C-OMP implementation of linear and nonlinear diffusion [1,2] for inpainting task (2D/3D case)
 * The minimisation is performed using explicit scheme. 
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. lambda - regularization parameter
 * 4. Edge-preserving parameter (sigma), when sigma equals to zero nonlinear diffusion -> linear diffusion
 * 5. Number of iterations, for explicit scheme >= 150 is recommended 
 * 6. tau - time-marching step for explicit scheme
 * 7. Penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
 *
 * Output:
 * [1] Inpainted image/volume 
 *
 * This function is based on the paper by
 * [1] Perona, P. and Malik, J., 1990. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), pp.629-639.
 * [2] Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.
 */

 
#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float Diffusion_Inpaint_CPU_main(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb,  float tau, int penaltytype, int dimX, int dimY, int dimZ);
CCPI_EXPORT float LinearDiff_Inp_2D(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float tau, int dimX, int dimY);
CCPI_EXPORT float NonLinearDiff_Inp_2D(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, float tau, int penaltytype, int dimX, int dimY);
CCPI_EXPORT float LinearDiff_Inp_3D(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float tau, int dimX, int dimY, int dimZ);
CCPI_EXPORT float NonLinearDiff_Inp_3D(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, float tau, int penaltytype, int dimX, int dimY, int dimZ);
#ifdef __cplusplus
}
#endif

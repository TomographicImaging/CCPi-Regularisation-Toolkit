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


/* C-OMP implementation of Nonlocal Vertical Marching inpainting method (2D case)
 * The method is heuristic but computationally efficent (especially for larger images).
 * It developed specifically to smoothly inpaint horizontal or inclined missing data regions in sinograms
 * The method WILL not work satisfactory if you have lengthy vertical stripes of missing data
 *
 * Inputs:
 * 1. 2D image or sinogram with horizontal or inclined regions of missing data
 * 2. Mask of the same size as A in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Linear increment to increase searching window size in iterations, values from 1-3 is a good choice

 * Output:
 * 1. Inpainted image or a sinogram
 * 2. updated mask
 *
 * Reference: TBA
 */

 
#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float NonlocalMarching_Inpaint_main(float *Input, unsigned char *M, float *Output, unsigned char *M_upd, int SW_increment, int iterationsNumb, int dimX, int dimY, int dimZ);
CCPI_EXPORT float inpaint_func(float *U, unsigned char *M_upd, float *Gauss_weights, int i, int j, int dimX, int dimY, int W_halfsize, int W_fullsize);
#ifdef __cplusplus
}
#endif

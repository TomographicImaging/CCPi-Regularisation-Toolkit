#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"
#include "CCPiDefines.h"

#define fTiny 0.00000001f
#define fLarge 100000000.0f
#define INFNORM -1

#define MAX(i,j) ((i)<(j) ? (j):(i))
#define MIN(i,j) ((i)<(j) ? (i):(j))

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

#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float TNV_CPU_main(float *Input, float *u, float lambda, int maxIter, float tol, int dimX, int dimY, int dimZ);

/*float PDHG(float *A, float *B, float tau, float sigma, float theta, float lambda, int p, int q, int r, float tol, int maxIter, int d_c, int d_w, int d_h);*/
CCPI_EXPORT float proxG(float *u_upd, float *v, float *f, float taulambda, long dimX, long dimY, long dimZ);
CCPI_EXPORT float gradient(float *u_upd, float *gradx_upd, float *grady_upd, long dimX, long dimY, long dimZ);
CCPI_EXPORT float proxF(float *gx, float *gy, float *vx, float *vy, float sigma, int p, int q, int r, long dimX, long dimY, long dimZ);
CCPI_EXPORT float divergence(float *qx_upd, float *qy_upd, float *div_upd, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif
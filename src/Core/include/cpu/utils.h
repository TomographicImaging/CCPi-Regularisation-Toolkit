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

#include <stdlib.h>
#include <memory.h>
#include "CCPiDefines.h"
#include "omp.h"
#ifdef __cplusplus
extern "C" {
#endif
CCPI_EXPORT float copyIm(float *A, float *U, long dimX, long dimY, long dimZ);
CCPI_EXPORT unsigned char copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ);
CCPI_EXPORT float copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher);
CCPI_EXPORT float TV_energy2D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY);
CCPI_EXPORT float TV_energy3D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY, int dimZ);
#ifdef __cplusplus
}
#endif

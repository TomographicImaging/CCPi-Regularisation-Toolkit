/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazanteev
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

#define _USE_MATH_DEFINES

#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY, int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop);
float PB_FUNC2D(float *A, float *B, int dimX, int dimY, int padXY, int SearchW, int SimilW, float h, float lambda);
float PB_FUNC3D(float *A, float *B, int dimX, int dimY, int dimZ, int padXY, int SearchW, int SimilW, float h, float lambda);

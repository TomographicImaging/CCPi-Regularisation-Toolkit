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

#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

#define EPS 0.01

/* 2D functions */
float der2D(float *U, float *D1, float *D2, int dimX, int dimY, int dimZ);
float div_upd2D(float *U0, float *U, float *D1, float *D2, int dimX, int dimY, int dimZ, float lambda, float tau);

float der3D(float *U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ);
float div_upd3D(float *U0, float *U, float *D1, float *D2, float *D3, unsigned short *Map, int switcher, int dimX, int dimY, int dimZ, float lambda, float tau);

float calcMap(float *U, unsigned short *Map, int dimX, int dimY, int dimZ);
float cleanMap(unsigned short *Map, int dimX, int dimY, int dimZ);

float copyIm(float *A, float *U, int dimX, int dimY, int dimZ);

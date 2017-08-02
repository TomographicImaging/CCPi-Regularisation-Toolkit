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

float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);
float gauss_seidel2D(float *U, float *A, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda, float mu);
float updDxDy_shrinkAniso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda);
float updDxDy_shrinkIso2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY, float lambda);
float updBxBy2D(float *U, float *Dx, float *Dy, float *Bx, float *By, int dimX, int dimY);

float gauss_seidel3D(float *U, float *A, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda, float mu);
float updDxDyDz_shrinkAniso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda);
float updDxDyDz_shrinkIso3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ, float lambda);
float updBxByBz3D(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int dimX, int dimY, int dimZ);

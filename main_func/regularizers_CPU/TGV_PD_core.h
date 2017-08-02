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
#include "utils.h"

/* 2D functions */
float DualP_2D(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, int dimZ, float sigma);
float ProjP_2D(float *P1, float *P2, int dimX, int dimY, int dimZ, float alpha1);
float DualQ_2D(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float sigma);
float ProjQ_2D(float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float alpha0);
float DivProjP_2D(float *U, float *A, float *P1, float *P2, int dimX, int dimY, int dimZ, float lambda, float tau);
float UpdV_2D(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, int dimZ, float tau);
float newU(float *U, float *U_old, int dimX, int dimY, int dimZ);
//float copyIm(float *A, float *U, int dimX, int dimY, int dimZ);

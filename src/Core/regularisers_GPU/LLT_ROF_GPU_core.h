#ifndef __ROFLLTGPU_H__
#define __ROFLLTGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT int LLT_ROF_GPU_main(float *Input, float *Output,  float *infovector, float lambdaROF, float lambdaLLT, int iterationsNumb, float tau,  float epsil, int gpu_device, int N, int M, int Z);

#endif

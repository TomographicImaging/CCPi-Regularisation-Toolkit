#ifndef __TVGPU_H__
#define __TVGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT int TV_ROF_GPU_main(float* Input, float* Output, float *infovector, float *lambdaPar, int lambda_is_arr, int iter, float tau, float epsil, int N, int M, int Z);

#endif

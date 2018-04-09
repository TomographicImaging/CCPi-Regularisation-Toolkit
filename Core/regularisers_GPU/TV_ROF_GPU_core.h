#ifndef __TVGPU_H__
#define __TVGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT void TV_ROF_GPU_main(float* Input, float* Output, float lambdaPar, int iter, float tau, int N, int M, int Z);

#endif 

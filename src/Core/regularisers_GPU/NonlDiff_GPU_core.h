#ifndef __NonlDiffGPU_H__
#define __NonlDiffGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT int NonlDiff_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, float epsil, int gpu_device, int N, int M, int Z);

#endif

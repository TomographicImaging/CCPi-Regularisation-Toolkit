#ifndef __NonlDiffGPU_H__
#define __NonlDiffGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT void NonlDiff_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int N, int M, int Z);

#endif 

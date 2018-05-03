#ifndef __Diff_4thO_GPU_H__
#define __Diff_4thO_GPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT void Diffus4th_GPU_main(float *Input, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int N, int M, int Z);

#endif 

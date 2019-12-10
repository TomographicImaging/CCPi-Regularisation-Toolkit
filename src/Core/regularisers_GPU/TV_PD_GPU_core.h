#ifndef _TV_PD_GPU_
#define _TV_PD_GPU_

#include "CCPiDefines.h"
#include <memory.h>

extern "C" CCPI_EXPORT int TV_PD_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil, float lipschitz_const, int methodTV, int nonneg, int dimX, int dimY, int dimZ);

#endif

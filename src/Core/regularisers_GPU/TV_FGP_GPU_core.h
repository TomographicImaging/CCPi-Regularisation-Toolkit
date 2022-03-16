#ifndef _TV_FGP_GPU_
#define _TV_FGP_GPU_

#include "CCPiDefines.h"
#include <memory.h>

extern "C" CCPI_EXPORT int TV_FGP_GPU_main(float *Input, float *Output, float *infovector, float lambdaPar, int iter, float epsil, int methodTV, int nonneg, int gpu_device, int dimX, int dimY, int dimZ);

#endif

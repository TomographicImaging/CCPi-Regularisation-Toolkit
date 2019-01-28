#ifndef _dTV_FGP_GPU_
#define _dTV_FGP_GPU_

#include "CCPiDefines.h"
#include <memory.h>

extern "C" CCPI_EXPORT int dTV_FGP_GPU_main(float *Input, float *InputRef, float *Output, float lambdaPar, int iter, float epsil, float eta, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);

#endif 

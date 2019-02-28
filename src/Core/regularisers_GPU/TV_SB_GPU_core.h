#ifndef _SB_TV_GPU_
#define _SB_TV_GPU_

#include "CCPiDefines.h"
#include <memory.h>


extern "C" CCPI_EXPORT int TV_SB_GPU_main(float *Input, float *Output, float mu, int iter, float epsil, int methodTV, int printM, int dimX, int dimY, int dimZ);

#endif 

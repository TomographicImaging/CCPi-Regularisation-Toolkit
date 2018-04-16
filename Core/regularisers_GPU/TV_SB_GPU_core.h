#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#ifndef _SB_TV_GPU_
#define _SB_TV_GPU_

extern "C" void TV_SB_GPU_main(float *Input, float *Output, float mu, int iter, float epsil, int methodTV, int printM, int dimX, int dimY, int dimZ);

#endif 

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#ifndef _TV_FGP_GPU_
#define _TV_FGP_GPU_

extern "C" void TV_FGP_GPU(float *Input, float *Output, float lambda, int iter, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);   

#endif 

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#ifndef _FGP_TV_GPU_
#define _FGP_TV_GPU_

extern "C" void FGP_TV_GPU(float *Input, float *Output, float lambda, int iter, float epsil, int methodTV, int nonneg, int printM, int dimX, int dimY, int dimZ);   

#endif 

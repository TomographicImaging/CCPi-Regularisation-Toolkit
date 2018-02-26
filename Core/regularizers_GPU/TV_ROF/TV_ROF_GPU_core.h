#ifndef __TVGPU_H__
#define __TVGPU_H__
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT void TV_ROF_GPU(float* Input, float* Output, int N, int M, int Z, int iter, float tau, float lambda);

#endif 

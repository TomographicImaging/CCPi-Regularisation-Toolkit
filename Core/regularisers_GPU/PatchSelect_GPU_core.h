#ifndef __NLREG_KERNELS_H_
#define __NLREG_KERNELS_H_
#include "CCPiDefines.h"
#include <stdio.h>

extern "C" CCPI_EXPORT void PatchSelect_GPU_main(float *A, unsigned short *H_i, unsigned short *H_j, float *Weights, int N, int M, int SearchWindow, int SimilarWin, int NumNeighb, float h);

#endif 

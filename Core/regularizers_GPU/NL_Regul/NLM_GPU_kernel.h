#ifndef __NLMREG_KERNELS_H_
#define __NLMREG_KERNELS_H_
#include "CCPiDefines.h"

extern "C" CCPI_EXPORT void NLM_GPU_kernel(float *A, float* B, float *Eucl_Vec, int N, int M, int Z, int dimension, int SearchW, int SimilW, int SearchW_real, float denh2, float lambda);
extern "C" CCPI_EXPORT float pad_crop(float *A, float *Ap, int OldSizeX, int OldSizeY, int OldSizeZ, int NewSizeX, int NewSizeY, int NewSizeZ, int padXY, int switchpad_crop);
#endif 

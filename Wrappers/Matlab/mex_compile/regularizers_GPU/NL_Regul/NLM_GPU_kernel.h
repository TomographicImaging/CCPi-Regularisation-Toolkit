#ifndef __NLMREG_KERNELS_H_
#define __NLMREG_KERNELS_H_

extern "C" void NLM_GPU_kernel(float *A, float* B, float *Eucl_Vec, int N, int M, int Z, int dimension, int SearchW, int SimilW, int SearchW_real, float denh2, float lambda);

#endif 

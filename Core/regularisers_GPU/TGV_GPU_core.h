#ifndef __TGV_GPU_H__
#define __TGV_GPU_H__

#include "CCPiDefines.h"
#include <memory.h>
#include <stdio.h>

extern "C" CCPI_EXPORT int TGV_GPU_main(float *U0, float *U, float lambda, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY, int dimZ);

#endif 

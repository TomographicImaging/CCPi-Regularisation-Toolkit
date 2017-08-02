/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazanteev
Copyright 2017 Srikanth Nagella, Edoardo Pasca

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "LLT_model_core.h"

float der2D(float *U, float *D1, float *D2, int dimX, int dimY, int dimZ)
{
	int i, j, i_p, i_m, j_m, j_p;
	float dxx, dyy, denom_xx, denom_yy;
#pragma omp parallel for shared(U,D1,D2) private(i, j, i_p, i_m, j_m, j_p, denom_xx, denom_yy, dxx, dyy)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* symmetric boundary conditions (Neuman) */
			i_p = i + 1; if (i_p == dimX) i_p = i - 1;
			i_m = i - 1; if (i_m < 0) i_m = i + 1;
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;

			dxx = U[i_p*dimY + j] - 2.0f*U[i*dimY + j] + U[i_m*dimY + j];
			dyy = U[i*dimY + j_p] - 2.0f*U[i*dimY + j] + U[i*dimY + j_m];

			denom_xx = fabs(dxx) + EPS;
			denom_yy = fabs(dyy) + EPS;

			D1[i*dimY + j] = dxx / denom_xx;
			D2[i*dimY + j] = dyy / denom_yy;
		}
	}
	return 1;
}
float div_upd2D(float *U0, float *U, float *D1, float *D2, int dimX, int dimY, int dimZ, float lambda, float tau)
{
	int i, j, i_p, i_m, j_m, j_p;
	float div, dxx, dyy;
#pragma omp parallel for shared(U,U0,D1,D2) private(i, j, i_p, i_m, j_m, j_p, div, dxx, dyy)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			/* symmetric boundary conditions (Neuman) */
			i_p = i + 1; if (i_p == dimX) i_p = i - 1;
			i_m = i - 1; if (i_m < 0) i_m = i + 1;
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;

			dxx = D1[i_p*dimY + j] - 2.0f*D1[i*dimY + j] + D1[i_m*dimY + j];
			dyy = D2[i*dimY + j_p] - 2.0f*D2[i*dimY + j] + D2[i*dimY + j_m];

			div = dxx + dyy;

			U[i*dimY + j] = U[i*dimY + j] - tau*div - tau*lambda*(U[i*dimY + j] - U0[i*dimY + j]);
		}
	}
	return *U0;
}

float der3D(float *U, float *D1, float *D2, float *D3, int dimX, int dimY, int dimZ)
{
	int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m;
	float dxx, dyy, dzz, denom_xx, denom_yy, denom_zz;
#pragma omp parallel for shared(U,D1,D2,D3) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, denom_xx, denom_yy, denom_zz, dxx, dyy, dzz)
	for (i = 0; i<dimX; i++) {
		/* symmetric boundary conditions (Neuman) */
		i_p = i + 1; if (i_p == dimX) i_p = i - 1;
		i_m = i - 1; if (i_m < 0) i_m = i + 1;
		for (j = 0; j<dimY; j++) {
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;
			for (k = 0; k<dimZ; k++) {
				k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
				k_m = k - 1; if (k_m < 0) k_m = k + 1;

				dxx = U[dimX*dimY*k + i_p*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i_m*dimY + j];
				dyy = U[dimX*dimY*k + i*dimY + j_p] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i*dimY + j_m];
				dzz = U[dimX*dimY*k_p + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m + i*dimY + j];

				denom_xx = fabs(dxx) + EPS;
				denom_yy = fabs(dyy) + EPS;
				denom_zz = fabs(dzz) + EPS;

				D1[dimX*dimY*k + i*dimY + j] = dxx / denom_xx;
				D2[dimX*dimY*k + i*dimY + j] = dyy / denom_yy;
				D3[dimX*dimY*k + i*dimY + j] = dzz / denom_zz;

			}
		}
	}
	return 1;
}

float div_upd3D(float *U0, float *U, float *D1, float *D2, float *D3, unsigned short *Map, int switcher, int dimX, int dimY, int dimZ, float lambda, float tau)
{
	int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m;
	float div, dxx, dyy, dzz;
#pragma omp parallel for shared(U,U0,D1,D2,D3) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, div, dxx, dyy, dzz)
	for (i = 0; i<dimX; i++) {
		/* symmetric boundary conditions (Neuman) */
		i_p = i + 1; if (i_p == dimX) i_p = i - 1;
		i_m = i - 1; if (i_m < 0) i_m = i + 1;
		for (j = 0; j<dimY; j++) {
			j_p = j + 1; if (j_p == dimY) j_p = j - 1;
			j_m = j - 1; if (j_m < 0) j_m = j + 1;
			for (k = 0; k<dimZ; k++) {
				k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
				k_m = k - 1; if (k_m < 0) k_m = k + 1;
				//                 k_p1 = k + 2; if (k_p1 >= dimZ) k_p1 = k - 2;
				//                 k_m1 = k - 2; if (k_m1 < 0) k_m1 = k + 2;   

				dxx = D1[dimX*dimY*k + i_p*dimY + j] - 2.0f*D1[dimX*dimY*k + i*dimY + j] + D1[dimX*dimY*k + i_m*dimY + j];
				dyy = D2[dimX*dimY*k + i*dimY + j_p] - 2.0f*D2[dimX*dimY*k + i*dimY + j] + D2[dimX*dimY*k + i*dimY + j_m];
				dzz = D3[dimX*dimY*k_p + i*dimY + j] - 2.0f*D3[dimX*dimY*k + i*dimY + j] + D3[dimX*dimY*k_m + i*dimY + j];

				if ((switcher == 1) && (Map[dimX*dimY*k + i*dimY + j] == 0)) dzz = 0;
				div = dxx + dyy + dzz;

				//                 if (switcher == 1) {                    
				// if (Map2[dimX*dimY*k + i*dimY + j] == 0) dzz2 = 0;
				//else dzz2 = D4[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*D4[dimX*dimY*k + i*dimY + j] + D4[dimX*dimY*k_m1 + i*dimY + j];
				//                     div = dzz + dzz2;
				//                 }

				//                 dzz = D3[dimX*dimY*k_p + i*dimY + j] - 2.0f*D3[dimX*dimY*k + i*dimY + j] + D3[dimX*dimY*k_m + i*dimY + j];
				//                 dzz2 = D4[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*D4[dimX*dimY*k + i*dimY + j] + D4[dimX*dimY*k_m1 + i*dimY + j];  
				//                 div = dzz + dzz2;

				U[dimX*dimY*k + i*dimY + j] = U[dimX*dimY*k + i*dimY + j] - tau*div - tau*lambda*(U[dimX*dimY*k + i*dimY + j] - U0[dimX*dimY*k + i*dimY + j]);
			}
		}
	}
	return *U0;
}

// float der3D_2(float *U, float *D1, float *D2, float *D3, float *D4, int dimX, int dimY, int dimZ)
// {
//     int i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, k_p1, k_m1;
//     float dxx, dyy, dzz, dzz2, denom_xx, denom_yy, denom_zz, denom_zz2;
// #pragma omp parallel for shared(U,D1,D2,D3,D4) private(i, j, k, i_p, i_m, j_m, j_p, k_p, k_m, denom_xx, denom_yy, denom_zz, denom_zz2, dxx, dyy, dzz, dzz2, k_p1, k_m1)
//     for(i=0; i<dimX; i++) {
//         /* symmetric boundary conditions (Neuman) */
//         i_p = i + 1; if (i_p == dimX) i_p = i - 1;
//         i_m = i - 1; if (i_m < 0) i_m = i + 1;
//         for(j=0; j<dimY; j++) {
//             j_p = j + 1; if (j_p == dimY) j_p = j - 1;
//             j_m = j - 1; if (j_m < 0) j_m = j + 1;
//             for(k=0; k<dimZ; k++) {
//                 k_p = k + 1; if (k_p == dimZ) k_p = k - 1;
//                 k_m = k - 1; if (k_m < 0) k_m = k + 1;
//                 k_p1 = k + 2; if (k_p1 >= dimZ) k_p1 = k - 2;
//                 k_m1 = k - 2; if (k_m1 < 0) k_m1 = k + 2;                
//                 
//                 dxx = U[dimX*dimY*k + i_p*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i_m*dimY + j];
//                 dyy = U[dimX*dimY*k + i*dimY + j_p] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k + i*dimY + j_m];
//                 dzz = U[dimX*dimY*k_p + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m + i*dimY + j];                
//                 dzz2 = U[dimX*dimY*k_p1 + i*dimY + j] - 2.0f*U[dimX*dimY*k + i*dimY + j] + U[dimX*dimY*k_m1 + i*dimY + j];                
//                 
//                 denom_xx = fabs(dxx) + EPS;
//                 denom_yy = fabs(dyy) + EPS;
//                 denom_zz = fabs(dzz) + EPS;
//                 denom_zz2 = fabs(dzz2) + EPS;
//                 
//                 D1[dimX*dimY*k + i*dimY + j] = dxx/denom_xx;
//                 D2[dimX*dimY*k + i*dimY + j] = dyy/denom_yy;
//                 D3[dimX*dimY*k + i*dimY + j] = dzz/denom_zz;               
//                 D4[dimX*dimY*k + i*dimY + j] = dzz2/denom_zz2;                               
//             }}}
//     return 1;
// }

float calcMap(float *U, unsigned short *Map, int dimX, int dimY, int dimZ)
{
	int i, j, k, i1, j1, i2, j2, windowSize;
	float val1, val2, thresh_val, maxval;
	windowSize = 1;
	thresh_val = 0.0001; /*thresh_val = 0.0035;*/

						 /* normalize volume first */
	maxval = 0.0f;
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {
				if (U[dimX*dimY*k + i*dimY + j] > maxval) maxval = U[dimX*dimY*k + i*dimY + j];
			}
		}
	}

	if (maxval != 0.0f) {
		for (i = 0; i<dimX; i++) {
			for (j = 0; j<dimY; j++) {
				for (k = 0; k<dimZ; k++) {
					U[dimX*dimY*k + i*dimY + j] = U[dimX*dimY*k + i*dimY + j] / maxval;
				}
			}
		}
	}
	else {
		printf("%s \n", "Maximum value is zero!");
		return 0;
	}

#pragma omp parallel for shared(U,Map) private(i, j, k, i1, j1, i2, j2, val1, val2)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {

				Map[dimX*dimY*k + i*dimY + j] = 0;
				//                 Map2[dimX*dimY*k + i*dimY + j] = 0; 

				val1 = 0.0f; val2 = 0.0f;
				for (i1 = -windowSize; i1 <= windowSize; i1++) {
					for (j1 = -windowSize; j1 <= windowSize; j1++) {
						i2 = i + i1;
						j2 = j + j1;

						if ((i2 >= 0) && (i2 < dimX) && (j2 >= 0) && (j2 < dimY)) {
							if (k == 0) {
								val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k + 1) + i2*dimY + j2], 2);
								//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);                                                  
							}
							else if (k == dimZ - 1) {
								val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k - 1) + i2*dimY + j2], 2);
								//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2);                           
							}
							//                       else if (k == 1) {
							//                           val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
							//                           val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);  
							//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);                           
							//                       }
							//                       else if (k == dimZ-2) {
							//                           val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-1) + i2*dimY + j2],2); 
							//                           val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+1) + i2*dimY + j2],2);      
							//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2);                           
							//                       }                      
							else {
								val1 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k - 1) + i2*dimY + j2], 2);
								val2 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k + 1) + i2*dimY + j2], 2);
								//                           val3 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k-2) + i2*dimY + j2],2); 
								//                           val4 += pow(U[dimX*dimY*k + i2*dimY + j2] - U[dimX*dimY*(k+2) + i2*dimY + j2],2);  
							}
						}
					}
				}

				val1 = 0.111f*val1; val2 = 0.111f*val2;
				//                  val3 = 0.111f*val3; val4 = 0.111f*val4;
				if ((val1 <= thresh_val) && (val2 <= thresh_val)) Map[dimX*dimY*k + i*dimY + j] = 1;
				//                  if ((val3 <= thresh_val) && (val4 <= thresh_val)) Map2[dimX*dimY*k + i*dimY + j] = 1;                        
			}
		}
	}
	return 1;
}

float cleanMap(unsigned short *Map, int dimX, int dimY, int dimZ)
{
	int i, j, k, i1, j1, i2, j2, counter;
#pragma omp parallel for shared(Map) private(i, j, k, i1, j1, i2, j2, counter)
	for (i = 0; i<dimX; i++) {
		for (j = 0; j<dimY; j++) {
			for (k = 0; k<dimZ; k++) {

				counter = 0;
				for (i1 = -3; i1 <= 3; i1++) {
					for (j1 = -3; j1 <= 3; j1++) {
						i2 = i + i1;
						j2 = j + j1;
						if ((i2 >= 0) && (i2 < dimX) && (j2 >= 0) && (j2 < dimY)) {
							if (Map[dimX*dimY*k + i2*dimY + j2] == 0) counter++;
						}
					}
				}
				if (counter < 24) Map[dimX*dimY*k + i*dimY + j] = 1;
			}
		}
	}
	return *Map;
}

/* Copy Image */
float copyIm(float *A, float *U, int dimX, int dimY, int dimZ)
{
	int j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return *U;
}
/*********************3D *********************/
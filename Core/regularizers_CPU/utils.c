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

#include "utils.h"

/* Copy Image */
float copyIm(float *A, float *U, int dimX, int dimY, int dimZ)
{
	int j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return *U;
}

/* function that calculates TV energy (ROF model) 
 * min||\nabla u|| + 0.5*lambda*||u -u0||^2
 * */
float TV_energy2D(float *U, float *U0, float *E_val, float lambda, int dimX, int dimY)
{
	int i, j, i1, j1, index;
	float NOMx_2, NOMy_2, E_Grad, E_Data;
	
	/* first calculate \grad U_xy*/	
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
				index = j*dimX+i;
                /* boundary conditions */
                i1 = i + 1; if (i == dimX-1) i1 = i;
                j1 = j + 1; if (j == dimY-1) j1 = j;
                
                /* Forward differences */
                NOMx_2 = pow(U[j1*dimX + i] - U[index],2); /* x+ */
                NOMy_2 = pow(U[j*dimX + i1] - U[index],2); /* y+ */
                E_Grad += sqrt(NOMx_2 + NOMy_2); /* gradient term energy */
                E_Data += 0.5f * lambda*(pow((U[index]-U0[index]),2)); /* fidelity term energy */
			}
		}
		E_val[0] = E_Grad + E_Data;
		return *E_val;
}

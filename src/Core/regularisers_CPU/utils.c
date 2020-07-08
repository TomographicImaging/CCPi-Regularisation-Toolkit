/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazanteev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils.h"
#include <math.h>

/* Copy Image (float) */
float copyIm(float *A, float *U, long dimX, long dimY, long dimZ)
{
    long j;
#pragma omp parallel for shared(A, U) private(j)
    for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
    return *U;
}

/* Copy Image */
unsigned char copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, U) private(j)
    for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
    return *U;
}

/*Roll image symmetrically from top to bottom*/
float copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher)
{
    int i, j;
#pragma omp parallel for shared(U, A) private(i,j)
    for (i=0; i<dimX; i++) {
        for (j=0; j<dimY; j++) {
            if (switcher == 0) {
                if (j < (dimY - roll_value)) U[j*dimX + i] = A[(j+roll_value)*dimX + i];
                else U[j*dimX + i] = A[(j - (dimY - roll_value))*dimX + i];
            }
            else {
                if (j < roll_value) U[j*dimX + i] = A[(j+(dimY - roll_value))*dimX + i];
                else U[j*dimX + i] = A[(j - roll_value)*dimX + i];
            }
        }}
    return *U;
}

int apply_nonnegativity(float *A, long DimTotal)
{
	long i;
#pragma omp parallel for shared(A) private(i)
	for (i = 0; i < DimTotal; i++)
	{
		A[i] = max(A[i], 0.0f);
	}

	return 1;
}

/* function that calculates TV energy
 * type - 1:  2*lambda*min||\nabla u|| + ||u -u0||^2
 * type - 2:  2*lambda*min||\nabla u||
 * */
float TV_energy2D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY)
{
    int i, j, i1, j1, index;
    float NOMx_2, NOMy_2, E_Grad=0.0f, E_Data=0.0f;

    /* first calculate \grad U_xy*/
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* boundary conditions */
            i1 = i + 1; if (i == dimX-1) i1 = i;
            j1 = j + 1; if (j == dimY-1) j1 = j;

            /* Forward differences */
            NOMx_2 = powf((float)(U[j1*dimX + i] - U[index]),2); /* x+ */
            NOMy_2 = powf((float)(U[j*dimX + i1] - U[index]),2); /* y+ */
            E_Grad += 2.0f*lambda*sqrtf((float)(NOMx_2) + (float)(NOMy_2)); /* gradient term energy */
            E_Data += powf((float)(U[index]-U0[index]),2); /* fidelity term energy */
        }
    }
    if (type == 1) E_val[0] = E_Grad + E_Data;
    if (type == 2) E_val[0] = E_Grad;
    return *E_val;
}

float TV_energy3D(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY, int dimZ)
{
    long i, j, k, i1, j1, k1, index;
    float NOMx_2, NOMy_2, NOMz_2, E_Grad=0.0f, E_Data=0.0f;

    /* first calculate \grad U_xy*/
    for(j=0; j<(long)(dimY); j++) {
        for(i=0; i<(long)(dimX); i++) {
            for(k=0; k<(long)(dimZ); k++) {
                index = (dimX*dimY)*k + j*dimX+i;
                /* boundary conditions */
                i1 = i + 1; if (i == (long)(dimX-1)) i1 = i;
                j1 = j + 1; if (j == (long)(dimY-1)) j1 = j;
                k1 = k + 1; if (k == (long)(dimZ-1)) k1 = k;

                /* Forward differences */
                NOMx_2 = powf((float)(U[(dimX*dimY)*k + j1*dimX+i] - U[index]),2); /* x+ */
                NOMy_2 = powf((float)(U[(dimX*dimY)*k + j*dimX+i1] - U[index]),2); /* y+ */
                NOMz_2 = powf((float)(U[(dimX*dimY)*k1 + j*dimX+i] - U[index]),2); /* z+ */

                E_Grad += 2.0f*lambda*sqrtf((float)(NOMx_2) + (float)(NOMy_2) + (float)(NOMz_2)); /* gradient term energy */
                E_Data += (powf((float)(U[index]-U0[index]),2)); /* fidelity term energy */
            }
        }
    }
    if (type == 1) E_val[0] = E_Grad + E_Data;
    if (type == 2) E_val[0] = E_Grad;
    return *E_val;
}

/* Down-Up scaling of 2D images using bilinear interpolation */
float Im_scale2D(float *Input, float *Scaled, int w, int h, int w2, int h2)
{
    int x, y, index, i, j;
    float x_ratio = ((float)(w-1))/w2;
    float y_ratio = ((float)(h-1))/h2;
    float A, B, C, D, x_diff, y_diff, gray;
    #pragma omp parallel for shared (Input, Scaled) private(x, y, index, A, B, C, D, x_diff, y_diff, gray)
    for (j=0;j<w2;j++) {
        for (i=0;i<h2;i++) {
            x = (int)(x_ratio * j);
            y = (int)(y_ratio * i);
            x_diff = (x_ratio * j) - x;
            y_diff = (y_ratio * i) - y;
            index = y*w+x ;

            A = Input[index];
            B = Input[index+1];
            C = Input[index+w];
            D = Input[index+w+1];

            gray = (float)(A*(1.0 - x_diff)*(1.0 - y_diff) +  B*(x_diff)*(1.0 - y_diff) +
                    C*(y_diff)*(1.0 - x_diff) +  D*(x_diff*y_diff));

            Scaled[i*w2+j] = gray;
        }}
    return *Scaled;
}

/*2D Projection onto convex set for P (called in PD_TV, FGP_dTV and FGP_TV methods)*/
float Proj_func2D(float *P1, float *P2, int methTV, long DimTotal)
{
    float val1, val2, denom, sq_denom;
    long i;
    if (methTV == 0)
	{
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,denom,sq_denom)
        for(i=0; i<DimTotal; i++)
		{
            denom = P1[i] * P1[i] + P2[i] * P2[i];
            if (denom > 1.0f)
			{
                sq_denom = 1.0f/sqrtf(denom);
                P1[i] *= sq_denom;
                P2[i] *= sq_denom;
            }
        }
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2) private(i,val1,val2)
        for(i=0; i<DimTotal; i++) {
            val1 = fabs(P1[i]);
            val2 = fabs(P2[i]);
            if (val1 < 1.0f) {val1 = 1.0f;}
            if (val2 < 1.0f) {val2 = 1.0f;}
            P1[i] = P1[i]/val1;
            P2[i] = P2[i]/val2;
        }
    }
    return 1;
}
/*3D Projection onto convex set for P (called in PD_TV, FGP_TV, FGP_dTV methods)*/
float Proj_func3D(float *P1, float *P2, float *P3, int methTV, long DimTotal)
{
    float val1, val2, val3, denom, sq_denom;
    long i;
    if (methTV == 0) {
        /* isotropic TV*/
#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3,sq_denom)
        for(i=0; i<DimTotal; i++) {
            denom = powf(P1[i],2) + powf(P2[i],2) + powf(P3[i],2);
            if (denom > 1.0f) {
                sq_denom = 1.0f/sqrtf(denom);
                P1[i] = P1[i]*sq_denom;
                P2[i] = P2[i]*sq_denom;
                P3[i] = P3[i]*sq_denom;
            }
        }
    }
    else {
        /* anisotropic TV*/
#pragma omp parallel for shared(P1,P2,P3) private(i,val1,val2,val3)
        for(i=0; i<DimTotal; i++) {
            val1 = fabs(P1[i]);
            val2 = fabs(P2[i]);
            val3 = fabs(P3[i]);
            if (val1 < 1.0f) {val1 = 1.0f;}
            if (val2 < 1.0f) {val2 = 1.0f;}
            if (val3 < 1.0f) {val3 = 1.0f;}
            P1[i] = P1[i]/val1;
            P2[i] = P2[i]/val2;
            P3[i] = P3[i]/val3;
        }
    }
    return 1;
}

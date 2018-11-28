#include "mex.h"
#include <matrix.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

#define EPS 1.0000e-12

/* C implementation of the spatial-dependent histogram
 * currently not optimal memory-wise
 *
 *
 * Input Parameters:
 * 1. 2D grayscale image (N x N)
 * 2. Number of histogram bins (M)
 * 4. Similarity window (half-size)
 *
 * Output:
 * 1. Filtered Image (N x N)
 *
 *
 * compile from Matlab with:
 * mex NLTV_SB_fast.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
 *
 * Im = double(imread('barb.bmp'))/255; % loading image
 * u0 = Im + .05*randn(size(Im)); u0(u0<0) = 0; % adding noise
 * [Filtered, theta, I1, J1] = NLTV_SB_fast(single(u0), 7, 7, 20, 0.1);
 * D. Kazantsev
 */

float copyIm(float *A, float *B, int dimX, int dimY, int dimZ);

/*2D functions */
float Indeces2D(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimY, int dimX, int NumNeighb, int SearchWindow,  int SimilarWin, float h2);
float NLM_ST_H1(float *Aorig, float *Output,  unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float beta, int IterNumb);



float denoise2D(float *Aorig, float *Output, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimY, int dimX, int NumNeighb);
/**************************************************/

void mexFunction(
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int number_of_dims, i, j, k, dimX, dimY, dimZ, SearchWindow, SimilarWin, NumNeighb,kk;
    unsigned short *H_i=NULL, *H_j=NULL;
    const int  *dim_array;
    float *A, *Output, *Weights, h, h2, lambda;
    int dim_array2[3];
    
    dim_array = mxGetDimensions(prhs[0]);
    number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    
    /*Handling Matlab input data*/
    A  = (float *) mxGetData(prhs[0]); /* a 2D image or a set of 2D images (3D stack) */
    SearchWindow = (int) mxGetScalar(prhs[1]);    /* Large Searching window to find and cluster intensities */
    SimilarWin = (int) mxGetScalar(prhs[2]);    /* Similarity window */
    NumNeighb = (int) mxGetScalar(prhs[3]); /* the total number of neighbours to take */
    h = (float) mxGetScalar(prhs[4]); /* NLM parameter */
                  
    h2 = h*h;    
    dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];
    dim_array2[0] = dimX; dim_array2[1] = dimY; dim_array2[2] = NumNeighb;  /* 2D case */ 
    
    /*****2D INPUT *****/
    if (number_of_dims == 2) {
        dimZ = 0;
        H_i = (unsigned short*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        H_j = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array2, mxUINT16_CLASS, mxREAL));
        Weights = (float*)mxGetPr(plhs[2] = mxCreateNumericArray(3, dim_array2, mxSINGLE_CLASS, mxREAL));
        Output = (float*)mxGetPr(plhs[3] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));        
       
    /* for each pixel store indeces of the most similar neighbours (patches) */
#pragma omp parallel for shared (A, Output, Weights, H_i, H_j) private(i,j)
       for(i=0; i<dimX; i++) {
            for(j=0; j<dimY; j++) {
                Indeces2D(A, H_i, H_j, Weights, i, j, dimX, dimY, NumNeighb, SearchWindow, SimilarWin, h2); 
                // denoise2D(A, Output, H_i, H_j, Weights, i, j, dimX, dimY, NumNeighb);
                NLM_ST_H1(A, Output, H_i, H_j, Weights, i, j, dimX, dimY, NumNeighb, 0.01f, 1);               
            }}          
    }
    /*****3D INPUT *****/
    /****************************************************/
    if (number_of_dims == 3) { 
	}
}


float Indeces2D(float *Aorig, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimY, int dimX, int NumNeighb, int SearchWindow, int SimilarWin, float h2)
{
    int i1, j1, i_m, j_m, i_c, j_c, i2, j2, i3, j3, k, counter, x, y;
    float *Weight_Vec, normsum, temp;
    unsigned short *ind_i, *ind_j, temp_i, temp_j;
    
   
    Weight_Vec = (float*) calloc((2*SearchWindow + 1)*(2*SearchWindow + 1), sizeof(float));
    ind_i = (unsigned short*) calloc((2*SearchWindow + 1)*(2*SearchWindow + 1), sizeof(unsigned short));
    ind_j = (unsigned short*) calloc((2*SearchWindow + 1)*(2*SearchWindow + 1), sizeof(unsigned short));    
    
         counter = 0;
        for(i_m=-SearchWindow; i_m<=SearchWindow; i_m++) {
            for(j_m=-SearchWindow; j_m<=SearchWindow; j_m++) {
                i1 = i+i_m;
                j1 = j+j_m;
                if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                        normsum = 0;
                        for(i_c=-SimilarWin; i_c<=SimilarWin; i_c++) {
                            for(j_c=-SimilarWin; j_c<=SimilarWin; j_c++) {
                                i2 = i1 + i_c;
                                j2 = j1 + j_c;
                                i3 = i + i_c;
                                j3 = j + j_c;
                                if (((i2 >= 0) && (i2 < dimX)) && ((j2 >= 0) && (j2 < dimY))) {
                                    if (((i3 >= 0) && (i3 < dimX)) && ((j3 >= 0) && (j3 < dimY))) {
                                        normsum += pow(Aorig[i3*dimY+j3] - Aorig[i2*dimY+j2], 2);
                                    }}
                            }}
                         /* writing temporarily into vectors */ 
                         if (normsum > EPS) Weight_Vec[counter] = exp(-normsum/h2);                      
                         ind_i[counter] = i1;
                         ind_j[counter] = j1;
                        counter ++;                    
                }
            }}
        /* do sorting to choose the most prominent weights [LOW -> HIGH]*/
        /* and re-arrange indeces accordingly */
         for(x=0; x < counter; x++)	{
             for(y=0; y < counter - 1; y++)		{
                 if(Weight_Vec[y] < Weight_Vec[y+1]) {
                     temp = Weight_Vec[y+1];
                     temp_i = ind_i[y+1];
                     temp_j = ind_j[y+1];
                     Weight_Vec[y+1] = Weight_Vec[y];
                     Weight_Vec[y] = temp;
                     ind_i[y+1] = ind_i[y];
                     ind_i[y] = temp_i;
                     ind_j[y+1] = ind_j[y];
                     ind_j[y] = temp_j;
                 }}} /*sorting loop end*/
         
 //       printf("%f %i %i \n", Weight_Vec[10], ind_i[10], ind_j[10]);
	 /*now select NumNeighb more prominent weights */
         for(x=0; x < NumNeighb; x++) {
             H_i[(dimX*dimY*x) + i*dimY+j] = ind_i[x];
             H_j[(dimX*dimY*x) + i*dimY+j] = ind_j[x];
             Weights[(dimX*dimY*x) + i*dimY+j] = Weight_Vec[x];            
         }
             
    free(ind_i);
    free(ind_j);
    free(Weight_Vec);
    return 1;
}

/* a test if NLM denoising works */
float denoise2D(float *Aorig, float *Output, unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimY, int dimX, int NumNeighb)
{
	int x, i1, j1; 
	float value = 0.0f, normweight  = 0.0f;
	
	for(x=0; x < NumNeighb; x++) {
		i1 = (H_i[(dimX*dimY*x) + i*dimY+j]);
		j1 = (H_j[(dimX*dimY*x) + i*dimY+j]);	
		value += Aorig[i1*dimY+j1]*Weights[(dimX*dimY*x) + i*dimY+j];
		normweight += Weights[(dimX*dimY*x) + i*dimY+j];
	}
	if (normweight != 0) Output[i*dimY+j] = value/normweight;
	else Output[i*dimY+j] = 0.0f;

	return *Output;
}

/***********<<<<Main Function for ST NLM - H1 penalty>>>>**********/
float NLM_ST_H1(float *Aorig, float *Output,  unsigned short *H_i, unsigned short *H_j, float *Weights, int i, int j, int dimX, int dimY, int NumNeighb, float beta, int IterNumb)
{
	int x, i1, j1; 
	float value = 0.0f, normweight  = 0.0f;
	
	for(x=0; x < NumNeighb; x++) {
		i1 = (H_i[(dimX*dimY*x) + i*dimY+j]);
		j1 = (H_j[(dimX*dimY*x) + i*dimY+j]);	
		value += Aorig[i1*dimY+j1]*Weights[(dimX*dimY*x) + i*dimY+j];
		normweight += Weights[(dimX*dimY*x) + i*dimY+j];
	}
	
//	if (normweight != 0) Output[i*dimY+j] = value/normweight;
//	else Output[i*dimY+j] = 0.0f;

    Output[i*dimY+j] = (beta*Aorig[i*dimY+j] + value)/(beta + normweight);
    return *Output;
}



/* General Functions */
/*****************************************************************/
/* Copy Image */
float copyIm(float *A, float *B, int dimX, int dimY, int dimZ)
{
    int j;
#pragma omp parallel for shared(A, B) private(j)
    for(j=0; j<dimX*dimY*dimZ; j++)  B[j] = A[j];
    return *B;
}

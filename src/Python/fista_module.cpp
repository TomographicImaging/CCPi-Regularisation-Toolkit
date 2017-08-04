/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazantsev
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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <cmath>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "boost/tuple/tuple.hpp"

#include "SplitBregman_TV_core.h"
#include "FGP_TV_core.h"
#include "LLT_model_core.h"
#include "utils.h"



#if defined(_WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(_WIN64)
#include <windows.h>
// this trick only if compiler is MSVC
__if_not_exists(uint8_t) { typedef __int8 uint8_t; }
__if_not_exists(uint16_t) { typedef __int8 uint16_t; }
#endif

namespace bp = boost::python;
namespace np = boost::python::numpy;

/*! in the Matlab implementation this is called as
void mexFunction(
int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[])
where:
prhs Array of pointers to the INPUT mxArrays
nrhs int number of INPUT mxArrays

nlhs Array of pointers to the OUTPUT mxArrays
plhs int number of OUTPUT mxArrays

***********************************************************

***********************************************************
double mxGetScalar(const mxArray *pm);
args: pm Pointer to an mxArray; cannot be a cell mxArray, a structure mxArray, or an empty mxArray.
Returns: Pointer to the value of the first real (nonimaginary) element of the mxArray.	In C, mxGetScalar returns a double.
***********************************************************
char *mxArrayToString(const mxArray *array_ptr);
args: array_ptr Pointer to mxCHAR array.
Returns: C-style string. Returns NULL on failure. Possible reasons for failure include out of memory and specifying an array that is not an mxCHAR array.
Description: Call mxArrayToString to copy the character data of an mxCHAR array into a C-style string.
***********************************************************
mxClassID mxGetClassID(const mxArray *pm);
args: pm Pointer to an mxArray
Returns: Numeric identifier of the class (category) of the mxArray that pm points to.For user-defined types,
mxGetClassId returns a unique value identifying the class of the array contents.
Use mxIsClass to determine whether an array is of a specific user-defined type.

mxClassID Value	  MATLAB Type   MEX Type	 C Primitive Type
mxINT8_CLASS 	  int8	        int8_T	     char, byte
mxUINT8_CLASS	  uint8	        uint8_T	     unsigned char, byte
mxINT16_CLASS	  int16	        int16_T	     short
mxUINT16_CLASS	  uint16	    uint16_T	 unsigned short
mxINT32_CLASS	  int32	        int32_T	     int
mxUINT32_CLASS	  uint32	    uint32_T	 unsigned int
mxINT64_CLASS	  int64	        int64_T	     long long
mxUINT64_CLASS	  uint64	    uint64_T 	 unsigned long long
mxSINGLE_CLASS	  single	    float	     float
mxDOUBLE_CLASS	  double	    double	     double

****************************************************************
double *mxGetPr(const mxArray *pm);
args: pm Pointer to an mxArray of type double
Returns: Pointer to the first element of the real data. Returns NULL in C (0 in Fortran) if there is no real data.
****************************************************************
mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims,
mxClassID classid, mxComplexity ComplexFlag);
args: ndimNumber of dimensions. If you specify a value for ndim that is less than 2, mxCreateNumericArray automatically sets the number of dimensions to 2.
dims Dimensions array. Each element in the dimensions array contains the size of the array in that dimension.
For example, in C, setting dims[0] to 5 and dims[1] to 7 establishes a 5-by-7 mxArray. Usually there are ndim elements in the dims array.
classid Identifier for the class of the array, which determines the way the numerical data is represented in memory.
For example, specifying mxINT16_CLASS in C causes each piece of numerical data in the mxArray to be represented as a 16-bit signed integer.
ComplexFlag  If the mxArray you are creating is to contain imaginary data, set ComplexFlag to mxCOMPLEX in C (1 in Fortran). Otherwise, set ComplexFlag to mxREAL in C (0 in Fortran).
Returns: Pointer to the created mxArray, if successful. If unsuccessful in a standalone (non-MEX file) application, returns NULL in C (0 in Fortran).
If unsuccessful in a MEX file, the MEX file terminates and returns control to the MATLAB prompt. The function is unsuccessful when there is not
enough free heap space to create the mxArray.
*/

void mexErrMessageText(char* text) {
	std::cerr << text << std::endl;
}

/*
double mxGetScalar(const mxArray *pm);
args: pm Pointer to an mxArray; cannot be a cell mxArray, a structure mxArray, or an empty mxArray.
Returns: Pointer to the value of the first real (nonimaginary) element of the mxArray.	In C, mxGetScalar returns a double.
*/

template<typename T>
double mxGetScalar(const np::ndarray plh) {
	return (double)bp::extract<T>(plh[0]);
}



template<typename T>
T * mxGetData(const np::ndarray pm) {
	//args: pm Pointer to an mxArray; cannot be a cell mxArray, a structure mxArray, or an empty mxArray.
	//Returns: Pointer to the value of the first real(nonimaginary) element of the mxArray.In C, mxGetScalar returns a double.
	/*Access the numpy array pointer:
	char * get_data() const;
	Returns:	Array’s raw data pointer as a char
	Note:	This returns char so stride math works properly on it.User will have to reinterpret_cast it.
	probably this would work.
	A = reinterpret_cast<float *>(prhs[0]);
	*/
	return reinterpret_cast<T *>(prhs[0]);
}

template<typename T>
np::ndarray zeros(int dims, int * dim_array, T el) {
	bp::tuple shape;
	if (dims == 3)
		shape = bp::make_tuple(dim_array[0], dim_array[1], dim_array[2]);
	else if (dims == 2)
		shape = bp::make_tuple(dim_array[0], dim_array[1]);
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray zz = np::zeros(shape, dtype);
	return zz;
}




bp::list mexFunction(np::ndarray input) {
	int number_of_dims = input.get_nd();
	int dim_array[3];

	dim_array[0] = input.shape(0);
	dim_array[1] = input.shape(1);
	if (number_of_dims == 2) {
		dim_array[2] = -1;
	}
	else {
		dim_array[2] = input.shape(2);
	}

	/**************************************************************************/
	np::ndarray zz = zeros(3, dim_array, (int)0);
	np::ndarray fzz = zeros(3, dim_array, (float)0);
	/**************************************************************************/

	int * A = reinterpret_cast<int *>(input.get_data());
	int * B = reinterpret_cast<int *>(zz.get_data());
	float * C = reinterpret_cast<float *>(fzz.get_data());

	//Copy data and cast
	for (int i = 0; i < dim_array[0]; i++) {
		for (int j = 0; j < dim_array[1]; j++) {
			for (int k = 0; k < dim_array[2]; k++) {
				int index = k + dim_array[2] * j + dim_array[2] * dim_array[1] * i;
				int val = (*(A + index));
				float fval = (float)val;
				std::memcpy(B + index, &val, sizeof(int));
				std::memcpy(C + index, &fval, sizeof(float));
			}
		}
	}

	bp::list result;

	result.append<int>(number_of_dims);
	result.append<int>(dim_array[0]);
	result.append<int>(dim_array[1]);
	result.append<int>(dim_array[2]);
	result.append<np::ndarray>(zz);
	result.append<np::ndarray>(fzz);

	//result.append<bp::tuple>(tup);
	return result;

}

bp::list SplitBregman_TV(np::ndarray input, double d_mu, int iter, double d_epsil, int methTV) {
	
	// the result is in the following list
	bp::list result;
		
	int number_of_dims, dimX, dimY, dimZ, ll, j, count;
	//const int  *dim_array;
	float *A, *U = NULL, *U_old = NULL, *Dx = NULL, *Dy = NULL, *Dz = NULL, *Bx = NULL, *By = NULL, *Bz = NULL, lambda, mu, epsil, re, re1, re_old;
	
	//number_of_dims = mxGetNumberOfDimensions(prhs[0]);
	//dim_array = mxGetDimensions(prhs[0]);

	number_of_dims = input.get_nd();
	int dim_array[3];

	dim_array[0] = input.shape(0);
	dim_array[1] = input.shape(1);
	if (number_of_dims == 2) {
		dim_array[2] = -1;
	}
	else {
		dim_array[2] = input.shape(2);
	}

	// Parameter handling is be done in Python
	///*Handling Matlab input data*/
	//if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");

	///*Handling Matlab input data*/
	//A = (float *)mxGetData(prhs[0]); /*noisy image (2D/3D) */
	A = reinterpret_cast<float *>(input.get_data());

	//mu = (float)mxGetScalar(prhs[1]); /* regularization parameter */
	mu = (float)d_mu;

	//iter = 35; /* default iterations number */
	
	//epsil = 0.0001; /* default tolerance constant */
	epsil = (float)d_epsil;
	//methTV = 0;  /* default isotropic TV penalty */
	//if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5))  iter = (int)mxGetScalar(prhs[2]); /* iterations number */
	//if ((nrhs == 4) || (nrhs == 5))  epsil = (float)mxGetScalar(prhs[3]); /* tolerance constant */
	//if (nrhs == 5) {
	//	char *penalty_type;
	//	penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
	//	if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
	//	if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
	//	mxFree(penalty_type);
	//}
	//if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { mexErrMsgTxt("The input image must be in a single precision"); }

	lambda = 2.0f*mu;
	count = 1;
	re_old = 0.0f;
	/*Handling Matlab output data*/
	dimY = dim_array[0]; dimX = dim_array[1]; dimZ = dim_array[2];

	if (number_of_dims == 2) {
		dimZ = 1; /*2D case*/
		//U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		//U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		//Dx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		//Dy = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		//Bx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		//By = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1]);
		np::dtype dtype = np::dtype::get_builtin<float>();

		np::ndarray npU = np::zeros(shape, dtype);
		np::ndarray npU_old = np::zeros(shape, dtype);
		np::ndarray npDx = np::zeros(shape, dtype);
		np::ndarray npDy = np::zeros(shape, dtype);
		np::ndarray npBx = np::zeros(shape, dtype);
		np::ndarray npBy = np::zeros(shape, dtype);

		U = reinterpret_cast<float *>(npU.get_data());
		U_old = reinterpret_cast<float *>(npU_old.get_data());
		Dx = reinterpret_cast<float *>(npDx.get_data());
		Dy = reinterpret_cast<float *>(npDy.get_data());
		Bx = reinterpret_cast<float *>(npBx.get_data());
		By = reinterpret_cast<float *>(npBy.get_data());



		copyIm(A, U, dimX, dimY, dimZ); /*initialize */

										/* begin outer SB iterations */
		for (ll = 0; ll < iter; ll++) {

			/*storing old values*/
			copyIm(U, U_old, dimX, dimY, dimZ);

			/*GS iteration */
			gauss_seidel2D(U, A, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);

			if (methTV == 1)  updDxDy_shrinkAniso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
			else updDxDy_shrinkIso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);

			updBxBy2D(U, Dx, Dy, Bx, By, dimX, dimY);

			/* calculate norm to terminate earlier */
			re = 0.0f; re1 = 0.0f;
			for (j = 0; j < dimX*dimY*dimZ; j++)
			{
				re += pow(U_old[j] - U[j], 2);
				re1 += pow(U_old[j], 2);
			}
			re = sqrt(re) / sqrt(re1);
			if (re < epsil)  count++;
			if (count > 4) break;

			/* check that the residual norm is decreasing */
			if (ll > 2) {
				if (re > re_old) break;
			}
			re_old = re;
			/*printf("%f %i %i \n", re, ll, count); */

			/*copyIm(U_old, U, dimX, dimY, dimZ); */
			
		}
		//printf("SB iterations stopped at iteration: %i\n", ll);
		result.append<np::ndarray>(npU);
		result.append<int>(ll);
	}
	if (number_of_dims == 3) {
			/*U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			Dx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			Dy = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			Dz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			Bx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			By = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
			Bz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));*/
			bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1], dim_array[2]);
			np::dtype dtype = np::dtype::get_builtin<float>();

			np::ndarray npU     = np::zeros(shape, dtype);
			np::ndarray npU_old = np::zeros(shape, dtype);
			np::ndarray npDx    = np::zeros(shape, dtype);
			np::ndarray npDy    = np::zeros(shape, dtype);
			np::ndarray npDz    = np::zeros(shape, dtype);
			np::ndarray npBx    = np::zeros(shape, dtype);
			np::ndarray npBy    = np::zeros(shape, dtype);
			np::ndarray npBz    = np::zeros(shape, dtype);

			U     = reinterpret_cast<float *>(npU.get_data());
			U_old = reinterpret_cast<float *>(npU_old.get_data());
			Dx    = reinterpret_cast<float *>(npDx.get_data());
			Dy    = reinterpret_cast<float *>(npDy.get_data());
			Dz    = reinterpret_cast<float *>(npDz.get_data());
			Bx    = reinterpret_cast<float *>(npBx.get_data());
			By    = reinterpret_cast<float *>(npBy.get_data());
			Bz    = reinterpret_cast<float *>(npBz.get_data());

			copyIm(A, U, dimX, dimY, dimZ); /*initialize */

											/* begin outer SB iterations */
			for (ll = 0; ll<iter; ll++) {

				/*storing old values*/
				copyIm(U, U_old, dimX, dimY, dimZ);

				/*GS iteration */
				gauss_seidel3D(U, A, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda, mu);

				if (methTV == 1) updDxDyDz_shrinkAniso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);
				else updDxDyDz_shrinkIso3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, lambda);

				updBxByBz3D(U, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ);

				/* calculate norm to terminate earlier */
				re = 0.0f; re1 = 0.0f;
				for (j = 0; j<dimX*dimY*dimZ; j++)
				{
					re += pow(U[j] - U_old[j], 2);
					re1 += pow(U[j], 2);
				}
				re = sqrt(re) / sqrt(re1);
				if (re < epsil)  count++;
				if (count > 4) break;

				/* check that the residual norm is decreasing */
				if (ll > 2) {
					if (re > re_old) break;
				}
				/*printf("%f %i %i \n", re, ll, count); */
				re_old = re;
			}
			//printf("SB iterations stopped at iteration: %i\n", ll);
			result.append<np::ndarray>(npU);
			result.append<int>(ll);
		}
	return result;

	}



bp::list FGP_TV(np::ndarray input, double d_mu, int iter, double d_epsil, int methTV) {

	// the result is in the following list
	bp::list result;

	int number_of_dims, dimX, dimY, dimZ, ll, j, count;
	float *A, *D = NULL, *D_old = NULL, *P1 = NULL, *P2 = NULL, *P3 = NULL, *P1_old = NULL, *P2_old = NULL, *P3_old = NULL, *R1 = NULL, *R2 = NULL, *R3 = NULL;
	float lambda, tk, tkp1, re, re1, re_old, epsil, funcval, mu;

	//number_of_dims = mxGetNumberOfDimensions(prhs[0]);
	//dim_array = mxGetDimensions(prhs[0]);

	number_of_dims = input.get_nd();
	int dim_array[3];

	dim_array[0] = input.shape(0);
	dim_array[1] = input.shape(1);
	if (number_of_dims == 2) {
		dim_array[2] = -1;
	}
	else {
		dim_array[2] = input.shape(2);
	}

	// Parameter handling is be done in Python
	///*Handling Matlab input data*/
	//if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");

	///*Handling Matlab input data*/
	//A = (float *)mxGetData(prhs[0]); /*noisy image (2D/3D) */
	A = reinterpret_cast<float *>(input.get_data());

	//mu = (float)mxGetScalar(prhs[1]); /* regularization parameter */
	mu = (float)d_mu;

	//iter = 35; /* default iterations number */

	//epsil = 0.0001; /* default tolerance constant */
	epsil = (float)d_epsil;
	//methTV = 0;  /* default isotropic TV penalty */
	//if ((nrhs == 3) || (nrhs == 4) || (nrhs == 5))  iter = (int)mxGetScalar(prhs[2]); /* iterations number */
	//if ((nrhs == 4) || (nrhs == 5))  epsil = (float)mxGetScalar(prhs[3]); /* tolerance constant */
	//if (nrhs == 5) {
	//	char *penalty_type;
	//	penalty_type = mxArrayToString(prhs[4]); /* choosing TV penalty: 'iso' or 'l1', 'iso' is the default */
	//	if ((strcmp(penalty_type, "l1") != 0) && (strcmp(penalty_type, "iso") != 0)) mexErrMsgTxt("Choose TV type: 'iso' or 'l1',");
	//	if (strcmp(penalty_type, "l1") == 0)  methTV = 1;  /* enable 'l1' penalty */
	//	mxFree(penalty_type);
	//}
	//if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { mexErrMsgTxt("The input image must be in a single precision"); }

	//plhs[1] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
	bp::tuple shape1 = bp::make_tuple(dim_array[0], dim_array[1]);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray out1 = np::zeros(shape1, dtype);
	
	//float *funcvalA = (float *)mxGetData(plhs[1]);
	float * funcvalA = reinterpret_cast<float *>(out1.get_data());
	//if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { mexErrMsgTxt("The input image must be in a single precision"); }

	/*Handling Matlab output data*/
	dimX = dim_array[0]; dimY = dim_array[1]; dimZ = dim_array[2];

	tk = 1.0f;
	tkp1 = 1.0f;
	count = 1;
	re_old = 0.0f;

	if (number_of_dims == 2) {
		dimZ = 1; /*2D case*/
		/*D = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		D_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		P1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		P2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		P1_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		P2_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		R1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		R2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));*/

		bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1]);
		np::dtype dtype = np::dtype::get_builtin<float>();


		np::ndarray npD      = np::zeros(shape, dtype);
		np::ndarray npD_old  = np::zeros(shape, dtype);
		np::ndarray npP1     = np::zeros(shape, dtype);
		np::ndarray npP2     = np::zeros(shape, dtype);
		np::ndarray npP1_old = np::zeros(shape, dtype);
		np::ndarray npP2_old = np::zeros(shape, dtype);
		np::ndarray npR1     = np::zeros(shape, dtype);
		np::ndarray npR2     = zeros(2, dim_array, (float)0);

		D      = reinterpret_cast<float *>(npD.get_data());
		D_old  = reinterpret_cast<float *>(npD_old.get_data());
		P1     = reinterpret_cast<float *>(npP1.get_data());
		P2     = reinterpret_cast<float *>(npP2.get_data());
		P1_old = reinterpret_cast<float *>(npP1_old.get_data());
		P2_old = reinterpret_cast<float *>(npP2_old.get_data());
		R1     = reinterpret_cast<float *>(npR1.get_data());
		R2     = reinterpret_cast<float *>(npR2.get_data());

		/* begin iterations */
		for (ll = 0; ll<iter; ll++) {

			/* computing the gradient of the objective function */
			Obj_func2D(A, D, R1, R2, lambda, dimX, dimY);

			/*Taking a step towards minus of the gradient*/
			Grad_func2D(P1, P2, D, R1, R2, lambda, dimX, dimY);

			/* projection step */
			Proj_func2D(P1, P2, methTV, dimX, dimY);

			/*updating R and t*/
			tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
			Rupd_func2D(P1, P1_old, P2, P2_old, R1, R2, tkp1, tk, dimX, dimY);

			/* calculate norm */
			re = 0.0f; re1 = 0.0f;
			for (j = 0; j<dimX*dimY*dimZ; j++)
			{
				re += pow(D[j] - D_old[j], 2);
				re1 += pow(D[j], 2);
			}
			re = sqrt(re) / sqrt(re1);
			if (re < epsil)  count++;
			if (count > 3) {
				Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);
				funcval = 0.0f;
				for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
				//funcvalA[0] = sqrt(funcval);
				float fv = sqrt(funcval);
				std::memcpy(funcvalA, &fv, sizeof(float));
				break;
			}

			/* check that the residual norm is decreasing */
			if (ll > 2) {
				if (re > re_old) {
					Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);
					funcval = 0.0f;
					for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
					//funcvalA[0] = sqrt(funcval);
					float fv = sqrt(funcval);
					std::memcpy(funcvalA, &fv, sizeof(float));
					break;
				}
			}
			re_old = re;
			/*printf("%f %i %i \n", re, ll, count); */

			/*storing old values*/
			copyIm(D, D_old, dimX, dimY, dimZ);
			copyIm(P1, P1_old, dimX, dimY, dimZ);
			copyIm(P2, P2_old, dimX, dimY, dimZ);
			tk = tkp1;

			/* calculating the objective function value */
			if (ll == (iter - 1)) {
				Obj_func2D(A, D, P1, P2, lambda, dimX, dimY);
				funcval = 0.0f;
				for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
				//funcvalA[0] = sqrt(funcval);
				float fv = sqrt(funcval);
				std::memcpy(funcvalA, &fv, sizeof(float));
			}
		}
		//printf("FGP-TV iterations stopped at iteration %i with the function value %f \n", ll, funcvalA[0]);
		result.append<np::ndarray>(npD);
		result.append<np::ndarray>(out1);
		result.append<int>(ll);
	}
	if (number_of_dims == 3) {
		/*D = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		D_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P1_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P2_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		P3_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		R1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		R2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		R3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));*/
		bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1], dim_array[2]);
		np::dtype dtype = np::dtype::get_builtin<float>();
		
		np::ndarray npD      = np::zeros(shape, dtype);
		np::ndarray npD_old  = np::zeros(shape, dtype);
		np::ndarray npP1     = np::zeros(shape, dtype);
		np::ndarray npP2     = np::zeros(shape, dtype);
		np::ndarray npP3     = np::zeros(shape, dtype);
		np::ndarray npP1_old = np::zeros(shape, dtype);
		np::ndarray npP2_old = np::zeros(shape, dtype);
		np::ndarray npP3_old = np::zeros(shape, dtype);
		np::ndarray npR1     = np::zeros(shape, dtype);
		np::ndarray npR2     = np::zeros(shape, dtype);
		np::ndarray npR3     = np::zeros(shape, dtype);

		D      = reinterpret_cast<float *>(npD.get_data());
		D_old  = reinterpret_cast<float *>(npD_old.get_data());
		P1     = reinterpret_cast<float *>(npP1.get_data());
		P2     = reinterpret_cast<float *>(npP2.get_data());
		P3     = reinterpret_cast<float *>(npP3.get_data());
		P1_old = reinterpret_cast<float *>(npP1_old.get_data());
		P2_old = reinterpret_cast<float *>(npP2_old.get_data());
		P3_old = reinterpret_cast<float *>(npP3_old.get_data());
		R1     = reinterpret_cast<float *>(npR1.get_data());
		R2     = reinterpret_cast<float *>(npR2.get_data());
		R2     = reinterpret_cast<float *>(npR3.get_data());
		/* begin iterations */
		for (ll = 0; ll<iter; ll++) {

			/* computing the gradient of the objective function */
			Obj_func3D(A, D, R1, R2, R3, lambda, dimX, dimY, dimZ);

			/*Taking a step towards minus of the gradient*/
			Grad_func3D(P1, P2, P3, D, R1, R2, R3, lambda, dimX, dimY, dimZ);

			/* projection step */
			Proj_func3D(P1, P2, P3, dimX, dimY, dimZ);

			/*updating R and t*/
			tkp1 = (1.0f + sqrt(1.0f + 4.0f*tk*tk))*0.5f;
			Rupd_func3D(P1, P1_old, P2, P2_old, P3, P3_old, R1, R2, R3, tkp1, tk, dimX, dimY, dimZ);

			/* calculate norm - stopping rules*/
			re = 0.0f; re1 = 0.0f;
			for (j = 0; j<dimX*dimY*dimZ; j++)
			{
				re += pow(D[j] - D_old[j], 2);
				re1 += pow(D[j], 2);
			}
			re = sqrt(re) / sqrt(re1);
			/* stop if the norm residual is less than the tolerance EPS */
			if (re < epsil)  count++;
			if (count > 3) {
				Obj_func3D(A, D, P1, P2, P3, lambda, dimX, dimY, dimZ);
				funcval = 0.0f;
				for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
				//funcvalA[0] = sqrt(funcval);
				float fv = sqrt(funcval);
				std::memcpy(funcvalA, &fv, sizeof(float));
				break;
			}

			/* check that the residual norm is decreasing */
			if (ll > 2) {
				if (re > re_old) {
					Obj_func3D(A, D, P1, P2, P3, lambda, dimX, dimY, dimZ);
					funcval = 0.0f;
					for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
					//funcvalA[0] = sqrt(funcval);
					float fv = sqrt(funcval);
					std::memcpy(funcvalA, &fv, sizeof(float));
					break;
				}
			}

			re_old = re;
			/*printf("%f %i %i \n", re, ll, count); */

			/*storing old values*/
			copyIm(D, D_old, dimX, dimY, dimZ);
			copyIm(P1, P1_old, dimX, dimY, dimZ);
			copyIm(P2, P2_old, dimX, dimY, dimZ);
			copyIm(P3, P3_old, dimX, dimY, dimZ);
			tk = tkp1;

			if (ll == (iter - 1)) {
				Obj_func3D(A, D, P1, P2, P3, lambda, dimX, dimY, dimZ);
				funcval = 0.0f;
				for (j = 0; j<dimX*dimY*dimZ; j++) funcval += pow(D[j], 2);
				//funcvalA[0] = sqrt(funcval);
				float fv = sqrt(funcval);
				std::memcpy(funcvalA, &fv, sizeof(float));
			}

		}
		//printf("FGP-TV iterations stopped at iteration %i with the function value %f \n", ll, funcvalA[0]);
		result.append<np::ndarray>(npD);
		result.append<np::ndarray>(out1);
		result.append<int>(ll);
	}

	return result;
}

bp::list LLT_model(np::ndarray input, double d_lambda, double d_tau, int iter, double d_epsil, int switcher) {
	// the result is in the following list
	bp::list result;

	int number_of_dims, dimX, dimY, dimZ, ll, j, count;
	//const int  *dim_array;
	float *U0, *U = NULL, *U_old = NULL, *D1 = NULL, *D2 = NULL, *D3 = NULL, lambda, tau, re, re1, epsil, re_old;
	unsigned short *Map = NULL;

	number_of_dims = input.get_nd();
	int dim_array[3];

	dim_array[0] = input.shape(0);
	dim_array[1] = input.shape(1);
	if (number_of_dims == 2) {
		dim_array[2] = -1;
	}
	else {
		dim_array[2] = input.shape(2);
	}

	///*Handling Matlab input data*/
	//U0 = (float *)mxGetData(prhs[0]); /*origanal noise image/volume*/
	//if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { mexErrMsgTxt("The input in single precision is required"); }
	//lambda = (float)mxGetScalar(prhs[1]); /*regularization parameter*/
	//tau = (float)mxGetScalar(prhs[2]); /* time-step */
	//iter = (int)mxGetScalar(prhs[3]); /*iterations number*/
	//epsil = (float)mxGetScalar(prhs[4]); /* tolerance constant */
	//switcher = (int)mxGetScalar(prhs[5]); /*switch on (1) restrictive smoothing in Z dimension*/
	
	U0 = reinterpret_cast<float *>(input.get_data());
	lambda = (float)d_lambda;
	tau = (float)d_tau;
	// iter is passed as parameter
	epsil = (float)d_epsil;
	// switcher is passed as parameter
										  /*Handling Matlab output data*/
	dimX = dim_array[0]; dimY = dim_array[1];  dimZ = 1;

	if (number_of_dims == 2) {
		/*2D case*/
		/*U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		D1 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		D2 = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));*/

		bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1]);
		np::dtype dtype = np::dtype::get_builtin<float>();


		np::ndarray npU = np::zeros(shape, dtype);
		np::ndarray npU_old = np::zeros(shape, dtype);
		np::ndarray npD1 = np::zeros(shape, dtype);
		np::ndarray npD2 = np::zeros(shape, dtype);
		

		U = reinterpret_cast<float *>(npU.get_data());
		U_old = reinterpret_cast<float *>(npU_old.get_data());
		D1 = reinterpret_cast<float *>(npD1.get_data());
		D2 = reinterpret_cast<float *>(npD2.get_data());
		
		/*Copy U0 to U*/
		copyIm(U0, U, dimX, dimY, dimZ);

		count = 1;
		re_old = 0.0f;

		for (ll = 0; ll < iter; ll++) {

			copyIm(U, U_old, dimX, dimY, dimZ);

			/*estimate inner derrivatives */
			der2D(U, D1, D2, dimX, dimY, dimZ);
			/* calculate div^2 and update */
			div_upd2D(U0, U, D1, D2, dimX, dimY, dimZ, lambda, tau);

			/* calculate norm to terminate earlier */
			re = 0.0f; re1 = 0.0f;
			for (j = 0; j<dimX*dimY*dimZ; j++)
			{
				re += pow(U_old[j] - U[j], 2);
				re1 += pow(U_old[j], 2);
			}
			re = sqrt(re) / sqrt(re1);
			if (re < epsil)  count++;
			if (count > 4) break;

			/* check that the residual norm is decreasing */
			if (ll > 2) {
				if (re > re_old) break;
			}
			re_old = re;

		} /*end of iterations*/
		  //printf("HO iterations stopped at iteration: %i\n", ll);

		result.append<np::ndarray>(npU);
	}
	else if (number_of_dims == 3) {
		/*3D case*/
		dimZ = dim_array[2];
		/*U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		D1 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		D2 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		D3 = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		if (switcher != 0) {
			Map = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array, mxUINT16_CLASS, mxREAL));
		}*/
		bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1]);
		np::dtype dtype = np::dtype::get_builtin<float>();


		np::ndarray npU = np::zeros(shape, dtype);
		np::ndarray npU_old = np::zeros(shape, dtype);
		np::ndarray npD1 = np::zeros(shape, dtype);
		np::ndarray npD2 = np::zeros(shape, dtype);
		np::ndarray npD3 = np::zeros(shape, dtype);
		np::ndarray npMap = np::zeros(shape, np::dtype::get_builtin<unsigned short>());
		Map = reinterpret_cast<unsigned short *>(npMap.get_data());
		if (switcher != 0) {
			//Map = (unsigned short*)mxGetPr(plhs[1] = mxCreateNumericArray(3, dim_array, mxUINT16_CLASS, mxREAL));
			
			Map = reinterpret_cast<unsigned short *>(npMap.get_data());
		}

		U = reinterpret_cast<float *>(npU.get_data());
		U_old = reinterpret_cast<float *>(npU_old.get_data());
		D1 = reinterpret_cast<float *>(npD1.get_data());
		D2 = reinterpret_cast<float *>(npD2.get_data());
		D3 = reinterpret_cast<float *>(npD2.get_data());
		
		/*Copy U0 to U*/
		copyIm(U0, U, dimX, dimY, dimZ);

		count = 1;
		re_old = 0.0f;
	

		if (switcher == 1) {
			/* apply restrictive smoothing */
			calcMap(U, Map, dimX, dimY, dimZ);
			/*clear outliers */
			cleanMap(Map, dimX, dimY, dimZ);
		}
		for (ll = 0; ll < iter; ll++) {

			copyIm(U, U_old, dimX, dimY, dimZ);

			/*estimate inner derrivatives */
			der3D(U, D1, D2, D3, dimX, dimY, dimZ);
			/* calculate div^2 and update */
			div_upd3D(U0, U, D1, D2, D3, Map, switcher, dimX, dimY, dimZ, lambda, tau);

			/* calculate norm to terminate earlier */
			re = 0.0f; re1 = 0.0f;
			for (j = 0; j<dimX*dimY*dimZ; j++)
			{
				re += pow(U_old[j] - U[j], 2);
				re1 += pow(U_old[j], 2);
			}
			re = sqrt(re) / sqrt(re1);
			if (re < epsil)  count++;
			if (count > 4) break;

			/* check that the residual norm is decreasing */
			if (ll > 2) {
				if (re > re_old) break;
			}
			re_old = re;

		} /*end of iterations*/
		//printf("HO iterations stopped at iteration: %i\n", ll);
		result.append<np::ndarray>(npU);
		if (switcher != 0) result.append<np::ndarray>(npMap);

	}
	return result;
}


BOOST_PYTHON_MODULE(regularizers)
{
	np::initialize();

	//To specify that this module is a package
	bp::object package = bp::scope();
	package.attr("__path__") = "regularizers";

	np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
	np::dtype dt2 = np::dtype::get_builtin<uint16_t>();

	def("mexFunction", mexFunction);
	def("SplitBregman_TV", SplitBregman_TV);
	def("FGP_TV", FGP_TV);
	def("LLT_model", LLT_model);
}
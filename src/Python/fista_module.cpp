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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <cmath>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "boost/tuple/tuple.hpp"

// include the regularizers
#include "FGP_TV_core.h"
#include "LLT_model_core.h"
#include "PatchBased_Regul_core.h"
#include "SplitBregman_TV_core.h"
#include "TGV_PD_core.h"

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
mxGetData
args: pm Pointer to an mxArray
Returns: Pointer to the first element of the real data. Returns NULL in C (0 in Fortran) if there is no real data.
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
mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims, mxClassID classid, mxComplexity ComplexFlag);
args:  ndim: Number of dimensions. If you specify a value for ndim that is less than 2, mxCreateNumericArray automatically sets the number of dimensions to 2.
       dims: Dimensions array. Each element in the dimensions array contains the size of the array in that dimension.
	         For example, in C, setting dims[0] to 5 and dims[1] to 7 establishes a 5-by-7 mxArray. Usually there are ndim elements in the dims array.
       classid: Identifier for the class of the array, which determines the way the numerical data is represented in memory.
                For example, specifying mxINT16_CLASS in C causes each piece of numerical data in the mxArray to be represented as a 16-bit signed integer.
       ComplexFlag:  If the mxArray you are creating is to contain imaginary data, set ComplexFlag to mxCOMPLEX in C (1 in Fortran). 
	                 Otherwise, set ComplexFlag to mxREAL in C (0 in Fortran).

Returns: Pointer to the created mxArray, if successful. If unsuccessful in a standalone (non-MEX file) application, returns NULL in C (0 in Fortran).
       If unsuccessful in a MEX file, the MEX file terminates and returns control to the MATLAB prompt. The function is unsuccessful when there is not
       enough free heap space to create the mxArray.
*/

template<typename T>
np::ndarray zeros(int dims, int * dim_array, T el) {
	bp::tuple shape = bp::make_tuple(dim_array[0], dim_array[1], dim_array[2]);
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray zz = np::zeros(shape, dtype);
	return zz;
}


bp::list SplitBregman_TV(np::ndarray input, double d_mu, , int niterations, double d_epsil, int TV_type) {
	/* C-OMP implementation of Split Bregman - TV denoising-regularization model (2D/3D)
	*
	* Input Parameters:
	* 1. Noisy image/volume
	* 2. lambda - regularization parameter
	* 3. Number of iterations [OPTIONAL parameter]
	* 4. eplsilon - tolerance constant [OPTIONAL parameter]
	* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
	*
	* Output:
	* Filtered/regularized image
	*
	* All sanity checks and default values are set in Python
	*/
	int number_of_dims, iter, dimX, dimY, dimZ, ll, j, count, methTV;
	const int dim_array[3];
	float *A, *U = NULL, *U_old = NULL, *Dx = NULL, *Dy = NULL, *Dz = NULL, *Bx = NULL, *By = NULL, *Bz = NULL, lambda, mu, epsil, re, re1, re_old;

	//number_of_dims = mxGetNumberOfDimensions(prhs[0]);
	//dim_array = mxGetDimensions(prhs[0]);
	number_of_dims = input.get_nd();

	dim_array[0] = input.shape(0);
	dim_array[1] = input.shape(1);
	if (number_of_dims == 2) {
		dim_array[2] = -11;
	}
	else {
		dim_array[2] = input.shape(2);
	}

	/*Handling Matlab input data*/
	//if ((nrhs < 2) || (nrhs > 5)) mexErrMsgTxt("At least 2 parameters is required: Image(2D/3D), Regularization parameter. The full list of parameters: Image(2D/3D), Regularization parameter, iterations number, tolerance, penalty type ('iso' or 'l1')");

	/*Handling Matlab input data*/
	//A = (float *)mxGetData(prhs[0]); /*noisy image (2D/3D) */
	A = reinterpret_cast<float *>(input.get_data());


	//mu = (float)mxGetScalar(prhs[1]); /* regularization parameter */
	mu = (float)d_mu;
	//iter = 35; /* default iterations number */
	iter = niterations;
	//epsil = 0.0001; /* default tolerance constant */
	epsil = (float)d_epsil;
	//methTV = 0;  /* default isotropic TV penalty */
	methTV = TV_type;
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
		/*
		mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims, mxClassID classid, mxComplexity ComplexFlag);
args:  ndim: Number of dimensions. If you specify a value for ndim that is less than 2, mxCreateNumericArray automatically sets the number of dimensions to 2.
       dims: Dimensions array. Each element in the dimensions array contains the size of the array in that dimension.
	         For example, in C, setting dims[0] to 5 and dims[1] to 7 establishes a 5-by-7 mxArray. Usually there are ndim elements in the dims array.
       classid: Identifier for the class of the array, which determines the way the numerical data is represented in memory.
                For example, specifying mxINT16_CLASS in C causes each piece of numerical data in the mxArray to be represented as a 16-bit signed integer.
       ComplexFlag:  If the mxArray you are creating is to contain imaginary data, set ComplexFlag to mxCOMPLEX in C (1 in Fortran). 
	                 Otherwise, set ComplexFlag to mxREAL in C (0 in Fortran).

					 mxCreateNumericArray initializes all its real data elements to 0.
*/

/*
		U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Dx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Dy = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Bx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		By = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
*/
		//U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		U = A = reinterpret_cast<float *>input.get_data();
		U_old = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Dx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Dy = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		Bx = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		By = (float*)mxGetPr(mxCreateNumericArray(2, dim_array, mxSINGLE_CLASS, mxREAL));
		copyIm(A, U, dimX, dimY, dimZ); /*initialize */

										/* begin outer SB iterations */
		for (ll = 0; ll<iter; ll++) {

			/*storing old values*/
			copyIm(U, U_old, dimX, dimY, dimZ);

			/*GS iteration */
			gauss_seidel2D(U, A, Dx, Dy, Bx, By, dimX, dimY, lambda, mu);

			if (methTV == 1)  updDxDy_shrinkAniso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);
			else updDxDy_shrinkIso2D(U, Dx, Dy, Bx, By, dimX, dimY, lambda);

			updBxBy2D(U, Dx, Dy, Bx, By, dimX, dimY);

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
			/*printf("%f %i %i \n", re, ll, count); */

			/*copyIm(U_old, U, dimX, dimY, dimZ); */
		}
		printf("SB iterations stopped at iteration: %i\n", ll);
	}
	if (number_of_dims == 3) {
		U = (float*)mxGetPr(plhs[0] = mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		U_old = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		Dx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		Dy = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		Dz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		Bx = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		By = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));
		Bz = (float*)mxGetPr(mxCreateNumericArray(3, dim_array, mxSINGLE_CLASS, mxREAL));

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
		printf("SB iterations stopped at iteration: %i\n", ll);
	}
	bp::list result;
	return result;
}
	

BOOST_PYTHON_MODULE(fista)
{
	np::initialize();

	//To specify that this module is a package
	bp::object package = bp::scope();
	package.attr("__path__") = "fista";

	np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
	np::dtype dt2 = np::dtype::get_builtin<uint16_t>();

	
	def("mexFunction", mexFunction);
}
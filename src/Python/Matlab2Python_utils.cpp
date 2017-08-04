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
np::ndarray zeros(int dims , int * dim_array, T el) {
	bp::tuple shape;
	if (dims == 3)
		shape = bp::make_tuple(dim_array[0], dim_array[1], dim_array[2]);
	else if (dims == 2)
		shape = bp::make_tuple(dim_array[0], dim_array[1]);
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray zz = np::zeros(shape, dtype);
	return zz;
}


bp::list mexFunction( np::ndarray input ) {
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
	
	int * A = reinterpret_cast<int *>( input.get_data() );
	int * B = reinterpret_cast<int *>( zz.get_data() );
	float * C = reinterpret_cast<float *>(fzz.get_data());

	//Copy data and cast
	for (int i = 0; i < dim_array[0]; i++) {
		for (int j = 0; j < dim_array[1]; j++) {
			for (int k = 0; k < dim_array[2]; k++) {
				int index = k + dim_array[2] * j + dim_array[2] * dim_array[1] * i;
				int val = (*(A + index));
				float fval = sqrt((float)val);
				std::memcpy(B + index , &val, sizeof(int));
				std::memcpy(C + index , &fval, sizeof(float));
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


BOOST_PYTHON_MODULE(prova)
{
	np::initialize();

	//To specify that this module is a package
	bp::object package = bp::scope();
	package.attr("__path__") = "fista";

	np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
	np::dtype dt2 = np::dtype::get_builtin<uint16_t>();
	
	//import_array();
	//numpy_boost_python_register_type<float, 1>();
	//numpy_boost_python_register_type<float, 2>();
	//numpy_boost_python_register_type<float, 3>();
	//numpy_boost_python_register_type<double, 3>();
	def("mexFunction", mexFunction);
}
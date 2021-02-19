/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazantsev
Copyright 2017 Srikanth Nagella, Edoardo Pasca
Copyright 2021 Gemma Fardell

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

#include <math.h>
#include "omp.h"
#include "CCPiDefines.h"

/* C-OMP templated Gradient functions
 */

#ifdef __cplusplus
extern "C" {
#endif
	CCPI_EXPORT void Grad_func3D_v2(float *P1, float *P2, float *P3, const float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ);
	CCPI_EXPORT void GradNorm_func3D_v2(const float *B, float *B_x, float *B_y, float *B_z, float eta, long dimX, long dimY, long dimZ);
	CCPI_EXPORT void Grad_func2D_v2(float *P1, float *P2, const float *D, const float *R1, const float *R2, float lambda, long dimX, long dimY);
	CCPI_EXPORT void GradNorm_func2D_v2(const float *B, float *B_x, float *B_y, float eta, long dimX, long dimY);
	CCPI_EXPORT void TV_energy3D_v2(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY, int dimZ);
	CCPI_EXPORT void TV_energy2D_v2(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY);

#ifdef __cplusplus
}
#endif

//Templated function for looping over a volume with boundary conditions at N
//inline functions must be difined in unique class
template <class T>
void gradient_direct_foward_3D(T *grad)
{
	long dimX = grad->get_dimX();
	long dimY = grad->get_dimY();
	long dimZ = grad->get_dimZ();
	long slice = dimX * dimY;
	long vol = slice * dimZ;

	long i, j, k, index;
	float val_x, val_y, val_z;

	for (k = 0; k < dimZ - 1; k++)
	{
#pragma omp parallel for private(i, j, index, val_x, val_y, val_z)
		for (j = 0; j < dimY - 1; j++)
		{
			index = k * slice + j * dimX;
			for (i = 0; i < dimX - 1; i++)
			{
				val_x = grad->get_val_x(index);
				val_y = grad->get_val_y(index);
				val_z = grad->get_val_z(index);
				grad->set_output_3D(index, val_x, val_y, val_z);

				index++;
			}

			val_x = grad->get_val_x_bc(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
		}

		index = k * slice + (dimY-1)*dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y_bc(index);
			val_z = grad->get_val_z(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
	}

	k = dimZ - 1;
#pragma omp parallel for private(i, j, index)
	for (j = 0; j < dimY - 1; j++)
	{
		index = k * slice + j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z_bc(index);
			grad->set_output_3D(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
	}

	index = vol - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		val_x = grad->get_val_x(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output_3D(index, val_x, val_y, val_z);
		index++;
	}

	val_x = grad->get_val_x_bc(index);
	val_y = grad->get_val_y_bc(index);
	val_z = grad->get_val_z_bc(index);
	grad->set_output_3D(index, val_x, val_y, val_z);
}
//Templated function for looping over a volume with boundary conditions at N
//inline functions must be difined in unique class
template <class T>
void gradient_direct_foward_2D(T *grad)
{
	long dimX = grad->get_dimX();
	long dimY = grad->get_dimY();
	long slice = dimX * dimY;

	long i, j, index;
	float val_x, val_y;

#pragma omp parallel for private(i, j, index, val_x, val_y)
	for (j = 0; j < dimY - 1; j++)
	{
		index = j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y(index);
			grad->set_output_2D(index, val_x, val_y);

			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y(index);
		grad->set_output_2D(index, val_x, val_y);
	}

	index = slice - dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		val_x = grad->get_val_x(index);
		val_y = grad->get_val_y_bc(index);
		grad->set_output_2D(index, val_x, val_y);
		index++;
	}

	val_x = grad->get_val_x_bc(index);
	val_y = grad->get_val_y_bc(index);
	grad->set_output_2D(index, val_x, val_y);
}

class base_gradient
{
protected:
	long m_dimX;
	long m_dimY;
	long m_dimZ;
	long m_slice;

public:
	inline long get_dimX()
	{
		return m_dimX;
	}
	inline long get_dimY()
	{
		return m_dimY;
	}
	inline long get_dimZ()
	{
		return m_dimZ;
	}

	static inline float FowardDifference(const float *arr, long index, long stride)
	{
		return arr[index] - arr[index + stride];
	}
};

class func : public base_gradient
{
private:
	float * m_P1;
	float * m_P2;
	float * m_P3;
	const float * m_D;
	const float * m_R1;
	const float * m_R2;
	const float * m_R3;
	float m_multip;


public:
	inline void set_output_2D(long index, float val_x, float val_y)
	{
		m_P1[index] = val_x;
		m_P2[index] = val_y;
	}
	inline void set_output_3D(long index, float val_x, float val_y, float val_z)
	{
		m_P1[index] = val_x;
		m_P2[index] = val_y;
		m_P3[index] = val_z;
	}
	inline float get_val_x(long index)
	{
		return m_R1[index] + m_multip * FowardDifference(m_D, index, 1);
	}
	inline float get_val_y(long index)
	{
		return m_R2[index] + m_multip * FowardDifference(m_D, index, m_dimX);
	}
	inline float get_val_z(long index)
	{
		return m_R3[index] + m_multip * FowardDifference(m_D, index, m_slice);
	}
	inline float get_val_x_bc(long index)
	{
		return m_R1[index];
	}
	inline float get_val_y_bc(long index)
	{
		return m_R2[index];
	}
	inline float get_val_z_bc(long index)
	{
		return m_R3[index];
	}
	func::func(float * P1, float * P2, float * P3, const float * D, const float * R1, const float * R2, const float * R3, float lambda, long dimX, long dimY, long dimZ)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;
		m_slice = dimX * dimY;

		m_P1 = P1;
		m_P2 = P2;
		m_P3 = P3;
		m_D = D;
		m_R1 = R1;
		m_R2 = R2;
		m_R3 = R3;
		m_multip = (1.0f / (26.0f*lambda)); //wrong for consistency
		//m_multip = (1.0f / (12.0f*lambda)); //different to FGP_TV::Grad_func3D
	}
	func::func(float * P1, float * P2, const float * D, const float * R1, const float * R2, float lambda, long dimX, long dimY)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_slice = dimX * dimY;

		m_P1 = P1;
		m_P2 = P2;
		m_D = D;
		m_R1 = R1;
		m_R2 = R2;
		m_multip = (1.0f / (8.0f*lambda));
	}
};

class GradNorm : public base_gradient
{
private:
	const float * m_B;
	float * m_Bx;
	float * m_By;
	float * m_Bz;
	float m_eta_sq;

public:
	inline void set_output_2D(long index, float val_x, float val_y)
	{
		float magn = val_x * val_x + val_y * val_y;
		float magn_inv = 1.0f / sqrtf(magn + m_eta_sq);

		m_Bx[index] = val_x * magn_inv;
		m_By[index] = val_y * magn_inv;
	}
	inline void set_output_3D(long index, float val_x, float val_y, float val_z)
	{
		float magn = val_x * val_x + val_y * val_y + val_z * val_z;
		float magn_inv = 1.0f / sqrtf(magn + m_eta_sq);

		m_Bx[index] = val_x * magn_inv;
		m_By[index] = val_y * magn_inv;
		m_Bz[index] = val_z * magn_inv;
	}
	inline float get_val_x(long index)
	{
		return -FowardDifference(m_B, index, 1);
	}
	inline float get_val_y(long index)
	{
		return -FowardDifference(m_B, index, m_dimX);
	}
	inline float get_val_z(long index)
	{
		return -FowardDifference(m_B, index, m_slice);
	}
	inline float get_val_x_bc(long index)
	{
		return -m_B[index];
	}
	inline float get_val_y_bc(long index)
	{
		return -m_B[index];
	}
	inline float get_val_z_bc(long index)
	{
		return -m_B[index];
	}

	GradNorm::GradNorm(const float *B, float *Bx, float *By, float *Bz, float eta, long dimX, long dimY, long dimZ)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;
		m_slice = dimX * dimY;

		m_B = B;
		m_Bx = Bx;
		m_By = By;
		m_Bz = Bz;
		m_eta_sq = eta * eta;
	}
	GradNorm::GradNorm(const float *B, float *Bx, float *By, float eta, long dimX, long dimY)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_slice = dimX * dimY;

		m_B = B;
		m_Bx = Bx;
		m_By = By;
		m_eta_sq = eta * eta;
	}
};
class TVenergy : public base_gradient
{
private:
	const float *m_U;
	const float *m_U0;
	float *m_E_grad_arr;
	float *m_E_data_arr;
	float m_lambda_2;


public:
	inline void set_output_2D(long index, float val_x, float val_y)
	{
		float fid = m_U[index] - m_U0[index];
		m_E_grad_arr[omp_get_thread_num()] += m_lambda_2*sqrtf(val_x * val_x + val_y * val_y);
		m_E_data_arr[omp_get_thread_num()] += fid * fid;
	}
	inline void set_output_3D(long index, float val_x, float val_y, float val_z)
	{
		float fid = m_U[index] - m_U0[index];
		m_E_grad_arr[omp_get_thread_num()] += m_lambda_2*sqrtf(val_x * val_x + val_y * val_y + val_z * val_z);
		m_E_data_arr[omp_get_thread_num()] += fid * fid;
	}
	inline float get_val_x(long index)
	{
		return FowardDifference(m_U, index, 1);
	}
	inline float get_val_y(long index)
	{
		return FowardDifference(m_U, index, m_dimX);
	}
	inline float get_val_z(long index)
	{
		return FowardDifference(m_U, index, m_slice);
	}
	inline float get_val_x_bc(long index)
	{
		return 0.f;
	}
	inline float get_val_y_bc(long index)
	{
		return 0.f;
	}
	inline float get_val_z_bc(long index)
	{
		return 0.f;
	}

	TVenergy::TVenergy(const float *U, const float *U0, float *E_grad_arr, float *E_data_arr, float lambda, int dimX, int dimY, int dimZ)
	{
		m_U = U;
		m_U0 = U0;
		m_E_grad_arr = E_grad_arr;
		m_E_data_arr = E_data_arr;
		m_lambda_2 = lambda * 2.f;
		m_dimX = dimX;
		m_dimY = dimX;
		m_dimZ = dimZ;

	}
	TVenergy::TVenergy(const float *U, const float *U0, float *E_grad_arr, float *E_data_arr, float lambda, int dimX, int dimY)
	{
		m_U = U;
		m_U0 = U0;
		m_E_grad_arr = E_grad_arr;
		m_E_data_arr = E_data_arr;
		m_lambda_2 = lambda * 2.f;
		m_dimX = dimX;
		m_dimY = dimX;
	}
};

class dTVenergy : public base_gradient
{
private:
	const float *m_U;
	const float *m_U0;
	float *m_E_grad_arr;
	float *m_E_data_arr;
	float m_lambda_2;

public:
	inline void set_output_2D(long index, float val_x, float val_y)
	{
		float fid = m_U[index] - m_U0[index];
		m_E_grad_arr[omp_get_thread_num()] += m_lambda_2*sqrtf(val_x * val_x + val_y * val_y);
		m_E_data_arr[omp_get_thread_num()] += fid * fid;
	}
	inline void set_output_3D(long index, float val_x, float val_y, float val_z)
	{
		float fid = m_U[index] - m_U0[index];
		m_E_grad_arr[omp_get_thread_num()] += m_lambda_2*sqrtf(val_x * val_x + val_y * val_y + val_z * val_z);
		m_E_data_arr[omp_get_thread_num()] += fid * fid;
	}
	inline float get_val_x(long index)
	{
		return FowardDifference(m_U, index, 1);
	}
	inline float get_val_y(long index)
	{
		return FowardDifference(m_U, index, m_dimX);
	}
	inline float get_val_z(long index)

	{
		return FowardDifference(m_U, index, m_slice);
	}
	inline float get_val_x_bc(long index)
	{
		return 0.f;
	}
	inline float get_val_y_bc(long index)
	{
		return 0.f;
	}
	inline float get_val_z_bc(long index)
	{
		return 0.f;
	}

	dTVenergy::dTVenergy(const float *U, const float *U0, float *E_grad_arr, float *E_data_arr, float lambda, int dimX, int dimY, int dimZ)
	{
		m_U = U;
		m_U0 = U0;
		m_E_grad_arr = E_grad_arr;
		m_E_data_arr = E_data_arr;
		m_lambda_2 = lambda * 2.f;
		m_dimX = dimX;
		m_dimY = dimX;
		m_dimZ = dimZ;
		m_slice = dimX * dimY;
	}
	dTVenergy::dTVenergy(const float *U, const float *U0, float *E_grad_arr, float *E_data_arr, float lambda, int dimX, int dimY)
	{
		m_U = U;
		m_U0 = U0;
		m_E_grad_arr = E_grad_arr;
		m_E_data_arr = E_data_arr;
		m_lambda_2 = lambda * 2.f;
		m_dimX = dimX;
		m_dimY = dimX;
		m_slice = dimX * dimY;
	}

};

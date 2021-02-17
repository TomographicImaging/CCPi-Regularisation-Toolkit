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

//Templated function for foward gradient type loop set_outputulations
//inline functions must be difined in unique class
template <class T>
void gradient_foward(T *grad)
{
	long dimX = grad->get_dimX();
	long dimY = grad->get_dimY();
	long dimZ = grad->get_dimZ();

	long i, j, k, index;
	float val_x, val_y, val_z;

	for (k = 0; k < dimZ - 1; k++)
	{
#pragma omp parallel for private(i, j, index, val_x, val_y, val_z)
		for (j = 0; j < dimY - 1; j++)
		{
			index = k * dimX * dimY + j * dimX;
			for (i = 0; i < dimX - 1; i++)
			{
				val_x = grad->get_val_x(index);
				val_y = grad->get_val_y(index);
				val_z = grad->get_val_z(index);
				grad->set_output(index, val_x, val_y, val_z);

				index++;
			}

			val_x = grad->get_val_x_bc(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z(index);
			grad->set_output(index, val_x, val_y, val_z);
		}

		j = dimY - 1;
		index = k * dimX * dimY + j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y_bc(index);
			val_z = grad->get_val_z(index);
			grad->set_output(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z(index);
		grad->set_output(index, val_x, val_y, val_z);
	}

	k = dimZ - 1;
#pragma omp parallel for private(i, j, index)
	for (j = 0; j < dimY - 1; j++)
	{
		index = k * dimX * dimY + j * dimX;
		for (i = 0; i < dimX - 1; i++)
		{
			val_x = grad->get_val_x(index);
			val_y = grad->get_val_y(index);
			val_z = grad->get_val_z(index);
			grad->set_output(index, val_x, val_y, val_z);
			index++;
		}

		val_x = grad->get_val_x_bc(index);
		val_y = grad->get_val_y(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output(index, val_x, val_y, val_z);
	}

	j = dimY - 1;
	index = k * dimX * dimY + j * dimX;
	for (i = 0; i < dimX - 1; i++)
	{
		val_x = grad->get_val_x(index);
		val_y = grad->get_val_y_bc(index);
		val_z = grad->get_val_z_bc(index);
		grad->set_output(index, val_x, val_y, val_z);
		index++;
	}

	val_x = grad->get_val_x_bc(index);
	val_y = grad->get_val_y_bc(index);
	val_z = grad->get_val_z_bc(index);
	grad->set_output(index, val_x, val_y, val_z);
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

class func_3D : public base_gradient
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
	inline void set_output(long index, float val_x, float val_y, float val_z)
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


	func_3D::func_3D(float * P1, float * P2, float * P3, const float * D, const float * R1, const float * R2, const float * R3, float lambda, long dimX, long dimY, long dimZ)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;

		m_P1 = P1;
		m_P2 = P2;
		m_P3 = P3;
		m_D = D;
		m_R1 = R1;
		m_R2 = R2;
		m_R3 = R3;
		m_multip = (1.0f / (12.0f*lambda));
	}

	func_3D::~func_3D() {};
};

class GradNorm_3D : public base_gradient
{
private:
	const float * m_B;
	float * m_Bx;
	float * m_By;
	float * m_Bz;
	float m_eta_sq;

public:
	inline void set_output(long index, float val_x, float val_y, float val_z)
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

	GradNorm_3D::GradNorm_3D(const float *B, float *Bx, float *By, float *Bz, float eta, long dimX, long dimY, long dimZ)
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

	GradNorm_3D::~GradNorm_3D() {};
};



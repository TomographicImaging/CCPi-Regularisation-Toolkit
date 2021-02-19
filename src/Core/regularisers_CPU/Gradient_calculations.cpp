#include <Gradient_calculations.h>
#include <iostream>

void Grad_func2D_v2(float *P1, float *P2, const float *D, const float *R1, const float *R2, float lambda, long dimX, long dimY)
{
	func grad = func(P1, P2, D, R1, R2, lambda, dimX, dimY);
	gradient_direct_foward_2D<func>(&grad);
}
void Grad_func3D_v2(float *P1, float *P2, float *P3, const float *D, const float *R1, const float *R2, const float *R3, float lambda, long dimX, long dimY, long dimZ)
{
	func grad = func(P1, P2, P3, D, R1, R2, R3, lambda, dimX, dimY, dimZ);
	gradient_direct_foward_3D<func>(&grad);
}
void GradNorm_func2D_v2(const float *B, float *B_x, float *B_y, float eta, long dimX, long dimY)
{
	GradNorm grad = GradNorm(B, B_x, B_y, eta, dimX, dimY);
	gradient_direct_foward_2D<GradNorm>(&grad);
}
void GradNorm_func3D_v2(const float *B, float *B_x, float *B_y, float *B_z, float eta, long dimX, long dimY, long dimZ)
{
	GradNorm grad = GradNorm(B, B_x, B_y, B_z, eta, dimX, dimY, dimZ);
	gradient_direct_foward_3D<GradNorm>(&grad);
}
void TV_energy3D_v2(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY, int dimZ)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

	float * E_grad_arr = new float[num_threads];
	memset(E_grad_arr, 0, num_threads * sizeof(float));

	float * E_data_arr = new float[omp_get_num_threads()];
	memset(E_data_arr, 0, num_threads * sizeof(float));

	TVenergy grad = TVenergy(U,  U0, E_grad_arr, E_data_arr, lambda, dimX, dimY, dimZ);
	gradient_direct_foward_3D<TVenergy>(&grad);

	float E_Grad = 0;
	float E_Data = 0;

	for (int i = 0; i < num_threads; i++)
	{
		E_Grad += E_grad_arr[i];
		E_Data += E_data_arr[i];
	}

	if (type == 1) E_val[0] = E_Grad + E_Data;
	if (type == 2) E_val[0] = E_Grad;

	delete[] E_grad_arr;
	delete[] E_data_arr;
}
void TV_energy2D_v2(float *U, float *U0, float *E_val, float lambda, int type, int dimX, int dimY)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

	float * E_grad_arr = new float[num_threads];
	memset(E_grad_arr, 0, num_threads * sizeof(float));

	float * E_data_arr = new float[omp_get_num_threads()];
	memset(E_data_arr, 0, num_threads * sizeof(float));

	TVenergy grad = TVenergy(U, U0, E_grad_arr, E_data_arr, lambda, dimX, dimY);
	gradient_direct_foward_2D<TVenergy>(&grad);

	float E_Grad = 0;
	float E_Data = 0;

	for (int i = 0; i < num_threads; i++)
	{
		E_Grad += E_grad_arr[i];
		E_Data += E_data_arr[i];
	}

	if (type == 1) E_val[0] = E_Grad + E_Data;
	if (type == 2) E_val[0] = E_Grad;

	delete[] E_grad_arr;
	delete[] E_data_arr;
}
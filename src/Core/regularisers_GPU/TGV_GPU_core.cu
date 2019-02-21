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

#include "TGV_GPU_core.h"
#include "shared.h"

/* CUDA implementation of Primal-Dual denoising method for 
 * Total Generilized Variation (TGV)-L2 model [1] (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume (2D/3D)
 * 2. lambda - regularisation parameter
 * 3. parameter to control the first-order term (alpha1)
 * 4. parameter to control the second-order term (alpha0)
 * 5. Number of Chambolle-Pock (Primal-Dual) iterations
 * 6. Lipshitz constant (default is 12)
 *
 * Output:
 * Filtered/regulariaed image 
 *
 * References:
 * [1] K. Bredies "Total Generalized Variation"
 */
    
#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8    
    
#define BLKXSIZE2D 16
#define BLKYSIZE2D 16
#define EPS 1.0e-7
#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void DualP_2D_kernel(float *U, float *V1, float *V2, float *P1, float *P2, int dimX, int dimY, float sigma)
{    
	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[index] += sigma*((U[j*dimX+(i-1)] - U[index]) - V1[index]); 
            else P1[index] += sigma*((U[j*dimX+(i+1)] - U[index])  - V1[index]); 
            if (j == dimY-1) P2[index] += sigma*((U[(j-1)*dimX+i] - U[index])  - V2[index]);
            else  P2[index] += sigma*((U[(j+1)*dimX+i] - U[index])  - V2[index]);
		}
	return;
} 

__global__ void ProjP_2D_kernel(float *P1, float *P2, int dimX, int dimY, float alpha1)
{
   	float grad_magn;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
            
            grad_magn = sqrt(pow(P1[index],2) + pow(P2[index],2));
            grad_magn = grad_magn/alpha1;
            if (grad_magn > 1.0f) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
            }
		}
	return;
} 

__global__ void DualQ_2D_kernel(float *V1, float *V2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, float sigma)
{
        float q1, q2, q11, q22;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {            
            /* symmetric boundary conditions (Neuman) */
    	    q1 = 0.0f; q11 = 0.0f; q2 = 0.0f; q22 = 0.0f;
            /* boundary conditions (Neuman) */
            if (i != dimX-1){
                q1 = V1[j*dimX+(i+1)] - V1[index];
                q11 = V2[j*dimX+(i+1)] - V2[index];
            }
            if (j != dimY-1) {
                q2 = V2[(j+1)*dimX+i] - V2[index];
                q22 = V1[(j+1)*dimX+i] - V1[index];
            }
            Q1[index] += sigma*(q1);
            Q2[index] += sigma*(q2);
            Q3[index] += sigma*(0.5f*(q11 + q22));
	}
	return;
} 

__global__ void ProjQ_2D_kernel(float *Q1, float *Q2, float *Q3, int dimX, int dimY, float alpha0)
{
	float grad_magn;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {            
            grad_magn = sqrt(pow(Q1[index],2) + pow(Q2[index],2) + 2*pow(Q3[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0f) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
        	    }
	}
	return;
} 

__global__ void DivProjP_2D_kernel(float *U, float *U0, float *P1, float *P2, int dimX, int dimY, float lambda, float tau)
{
	float P_v1, P_v2, div;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {
			
            if (i == 0) P_v1 = P1[index];
            else P_v1 = P1[index] - P1[j*dimX+(i-1)];
            if (j == 0) P_v2 = P2[index];
            else  P_v2 = P2[index] - P2[(j-1)*dimX+i];
            div = P_v1 + P_v2;
            U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);
		}
	return;
} 

__global__ void UpdV_2D_kernel(float *V1, float *V2, float *P1, float *P2, float *Q1, float *Q2, float *Q3, int dimX, int dimY, float tau)
{
	float q1, q3_x, q2, q3_y, div1, div2;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        
        int index = i + dimX*j;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY)) {			
   	    q2 = 0.0f;  q3_y = 0.0f; q1 = 0.0f; q3_x = 0.0;
            /* boundary conditions (Neuman) */
            if (i != 0) {
                q1 = Q1[index] - Q1[j*dimX+(i-1)];
                q3_x = Q3[index] - Q3[j*dimX+(i-1)];
            }
            if (j != 0) {
                q2 = Q2[index] - Q2[(j-1)*dimX+i];
                q3_y = Q3[index] - Q3[(j-1)*dimX+i];
            }
            div1 = q1 + q3_y;
            div2 = q3_x + q2;
            V1[index] += tau*(P1[index] + div1);
            V2[index] += tau*(P2[index] + div2);
	}
	return;
} 

__global__ void copyIm_TGV_kernel(float *U, float *U_old, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)   {
        U_old[index] = U[index];
    }
}

__global__ void copyIm_TGV_kernel_ar2(float *V1, float *V2, float *V1_old, float *V2_old, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)   {
        V1_old[index] = V1[index];
        V2_old[index] = V2[index];
    }
}

__global__ void newU_kernel(float *U, float *U_old, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        U[index] = 2.0f*U[index] - U_old[index];
    }
}


__global__ void newU_kernel_ar2(float *V1, float *V2, float *V1_old, float *V2_old, int N, int M, int num_total)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = xIndex + N*yIndex;
    
    if (index < num_total)	{
        V1[index] = 2.0f*V1[index] - V1_old[index];
        V2[index] = 2.0f*V2[index] - V2_old[index];  
    }
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
__global__ void DualP_3D_kernel(float *U, float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ, float sigma)
{    
	int index;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {
	
	    index = (dimX*dimY)*k + j*dimX+i;
            /* symmetric boundary conditions (Neuman) */
            if (i == dimX-1) P1[index] += sigma*((U[(dimX*dimY)*k + j*dimX+(i-1)] - U[index]) - V1[index]); 
            else P1[index] += sigma*((U[(dimX*dimY)*k + j*dimX+(i+1)] - U[index])  - V1[index]); 
            if (j == dimY-1) P2[index] += sigma*((U[(dimX*dimY)*k + (j-1)*dimX+i] - U[index])  - V2[index]);
            else  P2[index] += sigma*((U[(dimX*dimY)*k + (j+1)*dimX+i] - U[index])  - V2[index]);
            if (k == dimZ-1) P3[index] += sigma*((U[(dimX*dimY)*(k-1) + j*dimX+i] - U[index])  - V3[index]);
            else  P3[index] += sigma*((U[(dimX*dimY)*(k+1) + j*dimX+i] - U[index])  - V3[index]);
	}
	return;
} 

__global__ void ProjP_3D_kernel(float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ, float alpha1)
{
   	float grad_magn;
   	int index;
   	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {	
	    index = (dimX*dimY)*k + j*dimX+i;
            
            grad_magn = (sqrtf(pow(P1[index],2) + pow(P2[index],2) + pow(P3[index],2)))/alpha1;
            if (grad_magn > 1.0f) {
                P1[index] /= grad_magn;
                P2[index] /= grad_magn;
                P3[index] /= grad_magn;
            }
	}
	return;
}

__global__ void DualQ_3D_kernel(float *V1, float *V2, float *V3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, int dimX, int dimY, int dimZ, float sigma)
{
	int index; 
        float q1, q2, q3, q11, q22, q33, q44, q55, q66;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {	
	    
	    index = (dimX*dimY)*k + j*dimX+i;	    
	    q1 = 0.0f; q11 = 0.0f; q33 = 0.0f; q2 = 0.0f; q22 = 0.0f; q55 = 0.0f; q3 = 0.0f; q44 = 0.0f; q66 = 0.0f;
            /* symmetric boundary conditions (Neuman) */
            if (i != dimX-1){ 
                q1 = V1[(dimX*dimY)*k + j*dimX+(i+1)] - V1[index];              
                q11 = V2[(dimX*dimY)*k + j*dimX+(i+1)] - V2[index];
                q33 = V3[(dimX*dimY)*k + j*dimX+(i+1)] - V3[index];
            }
            if (j != dimY-1) {
                q2 = V2[(dimX*dimY)*k + (j+1)*dimX+i] - V2[index];                
                q22 = V1[(dimX*dimY)*k + (j+1)*dimX+i] - V1[index];
                q55 = V3[(dimX*dimY)*k + (j+1)*dimX+i] - V3[index];
            }
            if (k != dimZ-1) {
                q3 = V3[(dimX*dimY)*(k+1) + j*dimX+i] - V3[index];
                q44 = V1[(dimX*dimY)*(k+1) + j*dimX+i] - V1[index];
                q66 = V2[(dimX*dimY)*(k+1) + j*dimX+i] - V2[index];
            }
            
            Q1[index] += sigma*(q1); /*Q11*/
            Q2[index] += sigma*(q2); /*Q22*/            
            Q3[index] += sigma*(q3); /*Q33*/
            Q4[index] += sigma*(0.5f*(q11 + q22)); /* Q21 / Q12 */
            Q5[index] += sigma*(0.5f*(q33 + q44)); /* Q31 / Q13 */
            Q6[index] += sigma*(0.5f*(q55 + q66)); /* Q32 / Q23 */
	}
	return;
}


__global__ void ProjQ_3D_kernel(float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, int dimX, int dimY, int dimZ, float alpha0)
{
	float grad_magn;
	int index;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {	
	    
        index = (dimX*dimY)*k + j*dimX+i;	
	
	grad_magn = sqrtf(pow(Q1[index],2) + pow(Q2[index],2) + pow(Q3[index],2) + 2.0f*pow(Q4[index],2) + 2.0f*pow(Q5[index],2) + 2.0f*pow(Q6[index],2));
            grad_magn = grad_magn/alpha0;
            if (grad_magn > 1.0f) {
                Q1[index] /= grad_magn;
                Q2[index] /= grad_magn;
                Q3[index] /= grad_magn;
                Q4[index] /= grad_magn;
                Q5[index] /= grad_magn;
                Q6[index] /= grad_magn;
            }
	}
	return;
} 
__global__ void DivProjP_3D_kernel(float *U, float *U0, float *P1, float *P2, float *P3, int dimX, int dimY, int dimZ, float lambda, float tau)
{
	float P_v1, P_v2, P_v3, div;
	int index;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {	

        index = (dimX*dimY)*k + j*dimX+i;	
			
        if (i == 0) P_v1 = P1[index];
        else P_v1 = P1[index] - P1[(dimX*dimY)*k + j*dimX+(i-1)];
        if (j == 0) P_v2 = P2[index];
        else P_v2 = P2[index] - P2[(dimX*dimY)*k + (j-1)*dimX+i];
        if (k == 0) P_v3 = P3[index];
        else P_v3 = P3[index] - P3[(dimX*dimY)*(k-1) + (j)*dimX+i];              
                      
        div = P_v1 + P_v2 + P_v3;
        U[index] = (lambda*(U[index] + tau*div) + tau*U0[index])/(lambda + tau);             
	}
	return;
}
__global__ void UpdV_3D_kernel(float *V1, float *V2, float *V3, float *P1, float *P2, float *P3, float *Q1, float *Q2, float *Q3, float *Q4, float *Q5, float *Q6, int dimX, int dimY, int dimZ, float tau)
{
	float q1, q4x, q5x, q2, q4y, q6y, q6z, q5z, q3, div1, div2, div3;
	int index;
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;
        
        if ((i >= 0) && (i < dimX) && (j >= 0) && (j < dimY) && (k >= 0) && (k < dimZ)) {	

        index = (dimX*dimY)*k + j*dimX+i;	
        
	q1 = 0.0f; q4x= 0.0f; q5x= 0.0f; q2= 0.0f; q4y= 0.0f; q6y= 0.0f; q6z= 0.0f; q5z= 0.0f; q3= 0.0f;
        /* Q1 - Q11, Q2 - Q22, Q3 -  Q33, Q4 - Q21/Q12, Q5 - Q31/Q13, Q6 - Q32/Q23*/            
        /* symmetric boundary conditions (Neuman) */
        if (i != 0) {
                q1 = Q1[index] - Q1[(dimX*dimY)*k + j*dimX+(i-1)];
                q4x = Q4[index] - Q4[(dimX*dimY)*k + j*dimX+(i-1)];                
                q5x = Q5[index] - Q5[(dimX*dimY)*k + j*dimX+(i-1)];
        }
       if (j != 0) {
                q2 = Q2[index] - Q2[(dimX*dimY)*k + (j-1)*dimX+i];
                q4y = Q4[index] - Q4[(dimX*dimY)*k + (j-1)*dimX+i];
                q6y = Q6[index] - Q6[(dimX*dimY)*k + (j-1)*dimX+i];
       }
       if (k != 0) {
                q6z = Q6[index] - Q6[(dimX*dimY)*(k-1) + (j)*dimX+i];
                q5z = Q5[index] - Q5[(dimX*dimY)*(k-1) + (j)*dimX+i];
                q3 = Q3[index] - Q3[(dimX*dimY)*(k-1) + (j)*dimX+i];
       }
       div1 = q1 + q4y + q5z;
       div2 = q4x + q2 + q6z;            
       div3 = q5x + q6y + q3;
            
        V1[index] += tau*(P1[index] + div1);
        V2[index] += tau*(P2[index] + div2);
        V3[index] += tau*(P3[index] + div3);
	}
	return;
} 

__global__ void copyIm_TGV_kernel3D(float *U, float *U_old, int dimX, int dimY, int dimZ, int num_total)
{
    int index;
	
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;    
    
    index = (dimX*dimY)*k + j*dimX+i;
    
    if (index < num_total) {	
      	U_old[index] = U[index];	
    }
}

__global__ void copyIm_TGV_kernel3D_ar3(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, int dimX, int dimY, int dimZ, int num_total)
{
    int index;
	
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;    
    
    index = (dimX*dimY)*k + j*dimX+i;
    
    if (index < num_total) {	
      	V1_old[index] = V1[index];
	V2_old[index] = V2[index];
	V3_old[index] = V3[index];	
    }
}

__global__ void newU_kernel3D(float *U, float *U_old, int dimX, int dimY, int dimZ, int num_total)
{
     int index;
	
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     int j = blockDim.y * blockIdx.y + threadIdx.y;
     int k = blockDim.z * blockIdx.z + threadIdx.z;    
         
     index = (dimX*dimY)*k + j*dimX+i;
    
    if (index < num_total) {
	   U[index] = 2.0f*U[index] - U_old[index];
    }
}  

__global__ void newU_kernel3D_ar3(float *V1, float *V2, float *V3, float *V1_old, float *V2_old, float *V3_old, int dimX, int dimY, int dimZ, int num_total)
{
     int index;
	
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     int j = blockDim.y * blockIdx.y + threadIdx.y;
     int k = blockDim.z * blockIdx.z + threadIdx.z;    
         
     index = (dimX*dimY)*k + j*dimX+i;
    
    if (index < num_total) {
	   V1[index] = 2.0f*V1[index] - V1_old[index];
	   V2[index] = 2.0f*V2[index] - V2_old[index];
	   V3[index] = 2.0f*V3[index] - V3_old[index];
    }
}  

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/************************ MAIN HOST FUNCTION ***********************/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
extern "C" int TGV_GPU_main(float *U0, float *U, float lambda, float alpha1, float alpha0, int iterationsNumb, float L2, int dimX, int dimY, int dimZ)
{
	int dimTotal, dev = 0;
	CHECK(cudaSetDevice(dev));
	
	dimTotal = dimX*dimY*dimZ;
       
        float *U_old, *d_U0, *d_U, *P1, *P2, *Q1, *Q2, *Q3, *V1, *V1_old, *V2, *V2_old, tau, sigma;
        tau = pow(L2,-0.5);
        sigma = pow(L2,-0.5);
                                      
        CHECK(cudaMalloc((void**)&d_U0,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_U,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&U_old,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&P1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&P2,dimTotal*sizeof(float)));
        
        CHECK(cudaMalloc((void**)&Q1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q2,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V1,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V2,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V1_old,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V2_old,dimTotal*sizeof(float)));
        
        CHECK(cudaMemcpy(d_U0,U0,dimTotal*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_U,U0,dimTotal*sizeof(float),cudaMemcpyHostToDevice));      
        
        if (dimZ == 1) {
	/*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(dimX,BLKXSIZE2D), idivup(dimY,BLKYSIZE2D));
             
        for(int n=0; n < iterationsNumb; n++) {
			
	    /* Calculate Dual Variable P */
            DualP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, V1, V2, P1, P2, dimX, dimY, sigma);
	    CHECK(cudaDeviceSynchronize());
            /*Projection onto convex set for P*/
            ProjP_2D_kernel<<<dimGrid,dimBlock>>>(P1, P2, dimX, dimY, alpha1);
            CHECK(cudaDeviceSynchronize());
            /* Calculate Dual Variable Q */
            DualQ_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, Q1, Q2, Q3, dimX, dimY, sigma);
            CHECK(cudaDeviceSynchronize());
             /*Projection onto convex set for Q*/
            ProjQ_2D_kernel<<<dimGrid,dimBlock>>>(Q1, Q2, Q3, dimX, dimY, alpha0);
            CHECK(cudaDeviceSynchronize());
            /*saving U into U_old*/
            copyIm_TGV_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_2D_kernel<<<dimGrid,dimBlock>>>(d_U, d_U0, P1, P2, dimX, dimY, lambda, tau);
            CHECK(cudaDeviceSynchronize());
            /*get updated solution U*/
            newU_kernel<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*saving V into V_old*/
            copyIm_TGV_kernel_ar2<<<dimGrid,dimBlock>>>(V1, V2, V1_old, V2_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /* upd V*/
            UpdV_2D_kernel<<<dimGrid,dimBlock>>>(V1, V2, P1, P2, Q1, Q2, Q3, dimX, dimY, tau);
            CHECK(cudaDeviceSynchronize());
            /*get new V*/
            newU_kernel_ar2<<<dimGrid,dimBlock>>>(V1, V2, V1_old, V2_old, dimX, dimY, dimTotal);
            CHECK(cudaDeviceSynchronize());            
	        }
        }
        else {
        /*3D case */
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKYSIZE),idivup(dimZ,BLKXSIZE));
        
        float *P3, *Q4, *Q5, *Q6, *V3, *V3_old;
        
	CHECK(cudaMalloc((void**)&P3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q4,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q5,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&Q6,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V3,dimTotal*sizeof(float)));
        CHECK(cudaMalloc((void**)&V3_old,dimTotal*sizeof(float)));
        
        for(int n=0; n < iterationsNumb; n++) {
			
	    /* Calculate Dual Variable P */
            DualP_3D_kernel<<<dimGrid,dimBlock>>>(d_U, V1, V2, V3, P1, P2, P3, dimX, dimY, dimZ, sigma);
	    CHECK(cudaDeviceSynchronize());
            /*Projection onto convex set for P*/
            ProjP_3D_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, dimX, dimY, dimZ, alpha1);
            CHECK(cudaDeviceSynchronize());
            /* Calculate Dual Variable Q */
            DualQ_3D_kernel<<<dimGrid,dimBlock>>>(V1, V2, V3, Q1, Q2, Q3, Q4, Q5, Q6, dimX, dimY, dimZ, sigma);
            CHECK(cudaDeviceSynchronize());
             /*Projection onto convex set for Q*/
            ProjQ_3D_kernel<<<dimGrid,dimBlock>>>(Q1, Q2, Q3, Q4, Q5, Q6, dimX, dimY, dimZ, alpha0);
            CHECK(cudaDeviceSynchronize());
            /*saving U into U_old*/
            copyIm_TGV_kernel3D<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimZ, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*adjoint operation  -> divergence and projection of P*/
            DivProjP_3D_kernel<<<dimGrid,dimBlock>>>(d_U, d_U0, P1, P2, P3, dimX, dimY, dimZ, lambda, tau);
            CHECK(cudaDeviceSynchronize());
            /*get updated solution U*/
            newU_kernel3D<<<dimGrid,dimBlock>>>(d_U, U_old, dimX, dimY, dimZ, dimTotal);
            CHECK(cudaDeviceSynchronize());
            /*saving V into V_old*/
            copyIm_TGV_kernel3D_ar3<<<dimGrid,dimBlock>>>(V1, V2, V3, V1_old, V2_old, V3_old, dimX, dimY, dimZ, dimTotal);           
            CHECK(cudaDeviceSynchronize());
            /* upd V*/
            UpdV_3D_kernel<<<dimGrid,dimBlock>>>(V1, V2, V3, P1, P2, P3, Q1, Q2, Q3, Q4, Q5, Q6, dimX, dimY, dimZ, tau);
            CHECK(cudaDeviceSynchronize());
            /*get new V*/
            newU_kernel3D_ar3<<<dimGrid,dimBlock>>>(V1, V2, V3, V1_old, V2_old, V3_old, dimX, dimY, dimZ, dimTotal);
            CHECK(cudaDeviceSynchronize());            
	        }
	        
        CHECK(cudaFree(Q4));
        CHECK(cudaFree(Q5));
        CHECK(cudaFree(Q6));
        CHECK(cudaFree(P3));
        CHECK(cudaFree(V3));
        CHECK(cudaFree(V3_old));	                
        }
        
        CHECK(cudaMemcpy(U,d_U,dimTotal*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_U0));
        CHECK(cudaFree(d_U));
        CHECK(cudaFree(U_old));
        CHECK(cudaFree(P1));
        CHECK(cudaFree(P2));
        
        CHECK(cudaFree(Q1));
        CHECK(cudaFree(Q2));
        CHECK(cudaFree(Q3));
        CHECK(cudaFree(V1));
        CHECK(cudaFree(V2));
        CHECK(cudaFree(V1_old));
        CHECK(cudaFree(V2_old));
        return 0;
}

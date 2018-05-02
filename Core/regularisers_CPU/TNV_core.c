/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazantsev
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

#include "TNV_core.h"

/*
 * C-OMP implementation of Total Nuclear Variation regularisation model (2D + channels) [1]
 * The code is modified from the implementation by Joan Duran <joan.duran@uib.es> see
 * "denoisingPDHG_ipol.cpp" in Joans Collaborative Total Variation package
 *
 * Input Parameters:
 * 1. Noisy volume of 2D + channel dimension, i.e. 3D volume
 * 2. lambda - regularisation parameter
 * 3. Number of iterations [OPTIONAL parameter]
 * 4. eplsilon - tolerance constant [OPTIONAL parameter]
 * 5. print information: 0 (off) or 1 (on)  [OPTIONAL parameter]
 *
 * Output:
 * 1. Filtered/regularized image
 *
 * [1]. Duran, J., Moeller, M., Sbert, C. and Cremers, D., 2016. Collaborative total variation: a general framework for vectorial TV models. SIAM Journal on Imaging Sciences, 9(1), pp.116-151.
 */

float TNV_CPU_main(float *Input, float *u, float lambda, int maxIter, float tol, int dimX, int dimY, int dimZ)
{
    int k, p, q, r, DimTotal;
    float taulambda;
    float *u_upd, *gx, *gy, *gx_upd, *gy_upd, *qx, *qy, *qx_upd, *qy_upd, *v, *vx, *vy, *gradx, *grady, *gradx_upd, *grady_upd, *gradx_ubar, *grady_ubar, *div, *div_upd;
    
    p = 1;
    q = 1;
    r = 0;
    
    lambda = 1.0f/(2.0f*lambda);
    DimTotal = dimX*dimY*dimZ;
    /* PDHG algorithm parameters*/
    float tau = 0.5f;
    float sigma = 0.5f;
    float theta = 1.0f;
    
    // Auxiliar vectors
    u_upd = calloc(DimTotal, sizeof(float));
    gx = calloc(DimTotal, sizeof(float));
    gy = calloc(DimTotal, sizeof(float));
    gx_upd = calloc(DimTotal, sizeof(float));
    gy_upd = calloc(DimTotal, sizeof(float));
    qx = calloc(DimTotal, sizeof(float));
    qy = calloc(DimTotal, sizeof(float));
    qx_upd = calloc(DimTotal, sizeof(float));
    qy_upd = calloc(DimTotal, sizeof(float));
    v = calloc(DimTotal, sizeof(float));
    vx = calloc(DimTotal, sizeof(float));
    vy = calloc(DimTotal, sizeof(float));
    gradx = calloc(DimTotal, sizeof(float));
    grady = calloc(DimTotal, sizeof(float));
    gradx_upd = calloc(DimTotal, sizeof(float));
    grady_upd = calloc(DimTotal, sizeof(float));
    gradx_ubar = calloc(DimTotal, sizeof(float));
    grady_ubar = calloc(DimTotal, sizeof(float));
    div = calloc(DimTotal, sizeof(float));
    div_upd = calloc(DimTotal, sizeof(float));
    
    // Backtracking parameters
    float s = 1.0f;
    float gamma = 0.75f;
    float beta = 0.95f;
    float alpha0 = 0.2f;
    float alpha = alpha0;
    float delta = 1.5f;
    float eta = 0.95f;
    
    // PDHG algorithm parameters
    taulambda = tau * lambda;
    float divtau = 1.0f / tau;
    float divsigma = 1.0f / sigma;
    float theta1 = 1.0f + theta;
    
    /*allocate memory for  taulambda */
    //taulambda = (float*) calloc(dimZ, sizeof(float));
    //for(k=0; k < dimZ; k++)  {taulambda[k] = tau*lambda[k];}
    
    // Apply Primal-Dual Hybrid Gradient scheme
    int iter = 0;
    float residual = fLarge;
    float ubarx, ubary;
    
    for(iter = 0; iter < maxIter; iter++)   {
        // Argument of proximal mapping of fidelity term
#pragma omp parallel for shared(v, u) private(k)
        for(k=0; k<dimX*dimY*dimZ; k++)  {v[k] = u[k] + tau*div[k];}

// Proximal solution of fidelity term
proxG(u_upd, v, Input, taulambda, dimX, dimY, dimZ);

// Gradient of updated primal variable
gradient(u_upd, gradx_upd, grady_upd, dimX, dimY, dimZ);

// Argument of proximal mapping of regularization term
#pragma omp parallel for shared(gradx_upd, grady_upd, gradx, grady) private(k, ubarx, ubary)
for(k=0; k<dimX*dimY*dimZ; k++) {
    ubarx = theta1 * gradx_upd[k] - theta * gradx[k];
    ubary = theta1 * grady_upd[k] - theta * grady[k];
    vx[k] = ubarx + divsigma * qx[k];
    vy[k] = ubary + divsigma * qy[k];
    gradx_ubar[k] = ubarx;
    grady_ubar[k] = ubary;
}

proxF(gx_upd, gy_upd, vx, vy, sigma, p, q, r, dimX, dimY, dimZ);

// Update dual variable
#pragma omp parallel for shared(qx_upd, qy_upd) private(k)
for(k=0; k<dimX*dimY*dimZ; k++) {
    qx_upd[k] = qx[k] + sigma * (gradx_ubar[k] - gx_upd[k]);
    qy_upd[k] = qy[k] + sigma * (grady_ubar[k] - gy_upd[k]);
}

// Divergence of updated dual variable
#pragma omp parallel for shared(div_upd) private(k)
for(k=0; k<dimX*dimY*dimZ; k++)  {div_upd[k] = 0.0f;}
divergence(qx_upd, qy_upd, div_upd, dimX, dimY, dimZ);

// Compute primal residual, dual residual, and backtracking condition
float resprimal = 0.0f;
float resdual = 0.0f;
float product = 0.0f;
float unorm = 0.0f;
float qnorm = 0.0f;

for(k=0; k<dimX*dimY*dimZ; k++) {
    float udiff = u[k] - u_upd[k];
    float qxdiff = qx[k] - qx_upd[k];
    float qydiff = qy[k] - qy_upd[k];
    float divdiff = div[k] - div_upd[k];
    float gradxdiff = gradx[k] - gradx_upd[k];
    float gradydiff = grady[k] - grady_upd[k];
    
    resprimal += fabs(divtau*udiff + divdiff);
    resdual += fabs(divsigma*qxdiff - gradxdiff);
    resdual += fabs(divsigma*qydiff - gradydiff);
    
    unorm += (udiff * udiff);
    qnorm += (qxdiff * qxdiff + qydiff * qydiff);
    product += (gradxdiff * qxdiff + gradydiff * qydiff);
}

float b = (2.0f * tau * sigma * product) / (gamma * sigma * unorm +
        gamma * tau * qnorm);

// Adapt step-size parameters
float dual_dot_delta = resdual * s * delta;
float dual_div_delta = (resdual * s) / delta;

if(b > 1)
{
    // Decrease step-sizes to fit balancing principle
    tau = (beta * tau) / b;
    sigma = (beta * sigma) / b;
    alpha = alpha0;
    
    copyIm(u, u_upd, dimX, dimY, dimZ);
    copyIm(gx, gx_upd, dimX, dimY, dimZ);
    copyIm(gy, gy_upd, dimX, dimY, dimZ);
    copyIm(qx, qx_upd, dimX, dimY, dimZ);
    copyIm(qy, qy_upd, dimX, dimY, dimZ);
    copyIm(gradx, gradx_upd, dimX, dimY, dimZ);
    copyIm(grady, grady_upd, dimX, dimY, dimZ);
    copyIm(div, div_upd, dimX, dimY, dimZ);
    
} else if(resprimal > dual_dot_delta)
{
    // Increase primal step-size and decrease dual step-size
    tau = tau / (1.0f - alpha);
    sigma = sigma * (1.0f - alpha);
    alpha = alpha * eta;
    
} else if(resprimal < dual_div_delta)
{
    // Decrease primal step-size and increase dual step-size
    tau = tau * (1.0f - alpha);
    sigma = sigma / (1.0f - alpha);
    alpha = alpha * eta;
}

// Update variables
taulambda = tau * lambda;
//for(k=0; k < dimZ; k++) taulambda[k] = tau*lambda[k];

divsigma = 1.0f / sigma;
divtau = 1.0f / tau;

copyIm(u_upd, u, dimX, dimY, dimZ);
copyIm(gx_upd, gx, dimX, dimY, dimZ);
copyIm(gy_upd, gy, dimX, dimY, dimZ);
copyIm(qx_upd, qx, dimX, dimY, dimZ);
copyIm(qy_upd, qy, dimX, dimY, dimZ);
copyIm(gradx_upd, gradx, dimX, dimY, dimZ);
copyIm(grady_upd, grady, dimX, dimY, dimZ);
copyIm(div_upd, div, dimX, dimY, dimZ);

// Compute residual at current iteration
residual = (resprimal + resdual) / ((float) (dimX*dimY*dimZ));

//       printf("%f \n", residual);
if (residual < tol) {
    printf("Iterations stopped at %i with the residual %f \n", iter, residual);
    break; }

    }
    printf("Iterations stopped at %i with the residual %f \n", iter, residual);
    free (u_upd); free(gx); free(gy); free(gx_upd); free(gy_upd);
    free(qx); free(qy); free(qx_upd); free(qy_upd); free(v); free(vx); free(vy);
    free(gradx); free(grady); free(gradx_upd); free(grady_upd); free(gradx_ubar);
    free(grady_ubar); free(div); free(div_upd);
    
    return *u;
}

float proxG(float *u_upd, float *v, float *f, float taulambda, int dimX, int dimY, int dimZ)
{
    float constant;
    int k;
    constant = 1.0f + taulambda;
#pragma omp parallel for shared(v, f, u_upd) private(k)
    for(k=0; k<dimZ*dimX*dimY; k++) {
        u_upd[k] = (v[k] + taulambda * f[k])/constant;
        //u_upd[(dimX*dimY)*k + l] = (v[(dimX*dimY)*k + l] + taulambda * f[(dimX*dimY)*k + l])/constant;
    }
    return *u_upd;
}

float gradient(float *u_upd, float *gradx_upd, float *grady_upd, int dimX, int dimY, int dimZ)
{
    int i, j, k, l;
    // Compute discrete gradient using forward differences
#pragma omp parallel for shared(gradx_upd,grady_upd,u_upd) private(i, j, k, l)
    for(k = 0; k < dimZ; k++)   {
        for(j = 0; j < dimY; j++)   {
            l = j * dimX;           
            for(i = 0; i < dimX; i++)   {
                // Derivatives in the x-direction
                if(i != dimX-1)
                    gradx_upd[(dimX*dimY)*k + i+l] = u_upd[(dimX*dimY)*k + i+1+l] - u_upd[(dimX*dimY)*k + i+l];
                else
                    gradx_upd[(dimX*dimY)*k + i+l] = 0.0f;
                
                // Derivatives in the y-direction
                if(j != dimY-1)
                    //grady_upd[(dimX*dimY)*k + i+l] = u_upd[(dimX*dimY)*k + i+dimY+l] -u_upd[(dimX*dimY)*k + i+l];
                    grady_upd[(dimX*dimY)*k + i+l] = u_upd[(dimX*dimY)*k + i+(j+1)*dimX] -u_upd[(dimX*dimY)*k + i+l];
                else
                    grady_upd[(dimX*dimY)*k + i+l] = 0.0f;
            }}}
    return 1;
}

float proxF(float *gx, float *gy, float *vx, float *vy, float sigma, int p, int q, int r, int dimX, int dimY, int dimZ)
{
    // (S^p, \ell^1) norm decouples at each pixel
//   Spl1(gx, gy, vx, vy, sigma, p, num_channels, dim);
    float divsigma = 1.0f / sigma;
    
    // $\ell^{1,1,1}$-TV regularization
//       int i,j,k;
//     #pragma omp parallel for shared (gx,gy,vx,vy) private(i,j,k)
//      for(k = 0; k < dimZ; k++)  {
//         for(i=0; i<dimX; i++) {
//              for(j=0; j<dimY; j++) {
//                 gx[(dimX*dimY)*k + (i)*dimY + (j)] = SIGN(vx[(dimX*dimY)*k + (i)*dimY + (j)]) * MAX(fabs(vx[(dimX*dimY)*k + (i)*dimY + (j)]) - divsigma,  0.0f);
//                 gy[(dimX*dimY)*k + (i)*dimY + (j)] = SIGN(vy[(dimX*dimY)*k + (i)*dimY + (j)]) * MAX(fabs(vy[(dimX*dimY)*k + (i)*dimY + (j)]) - divsigma,  0.0f);
//             }}}
    
    // Auxiliar vector
    float *proj, sum, shrinkfactor ;
    float M1,M2,M3,valuex,valuey,T,D,det,eig1,eig2,sig1,sig2,V1, V2, V3, V4, v0,v1,v2, mu1,mu2,sig1_upd,sig2_upd,t1,t2,t3;
    int i,j,k, ii, num;
#pragma omp parallel for shared (gx,gy,vx,vy,p) private(i,ii,j,k,proj,num, sum, shrinkfactor, M1,M2,M3,valuex,valuey,T,D,det,eig1,eig2,sig1,sig2,V1, V2, V3, V4,v0,v1,v2,mu1,mu2,sig1_upd,sig2_upd,t1,t2,t3)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            
            proj = (float*) calloc (2,sizeof(float));
            // Compute matrix $M\in\R^{2\times 2}$
            M1 = 0.0f;
            M2 = 0.0f;
            M3 = 0.0f;
            
            for(k = 0; k < dimZ; k++)
            {
                valuex = vx[(dimX*dimY)*k + (j)*dimX + (i)];
                valuey = vy[(dimX*dimY)*k + (j)*dimX + (i)];
                
                M1 += (valuex * valuex);
                M2 += (valuex * valuey);
                M3 += (valuey * valuey);
            }
            
            // Compute eigenvalues of M
            T = M1 + M3;
            D = M1 * M3 - M2 * M2;
            det = sqrt(MAX((T * T / 4.0f) - D, 0.0f));
            eig1 = MAX((T / 2.0f) + det, 0.0f);
            eig2 = MAX((T / 2.0f) - det, 0.0f);
            sig1 = sqrt(eig1);
            sig2 = sqrt(eig2);
            
            // Compute normalized eigenvectors
            V1 = V2 = V3 = V4 = 0.0f;
            
            if(M2 != 0.0f)
            {
                v0 = M2;
                v1 = eig1 - M3;
                v2 = eig2 - M3;
                
                mu1 = sqrtf(v0 * v0 + v1 * v1);
                mu2 = sqrtf(v0 * v0 + v2 * v2);
                
                if(mu1 > fTiny)
                {
                    V1 = v1 / mu1;
                    V3 = v0 / mu1;
                }
                
                if(mu2 > fTiny)
                {
                    V2 = v2 / mu2;
                    V4 = v0 / mu2;
                }
                
            } else
            {
                if(M1 > M3)
                {
                    V1 = V4 = 1.0f;
                    V2 = V3 = 0.0f;
                    
                } else
                {
                    V1 = V4 = 0.0f;
                    V2 = V3 = 1.0f;
                }
            }
            
            // Compute prox_p of the diagonal entries
            sig1_upd = sig2_upd = 0.0f;
            
            if(p == 1)
            {
                sig1_upd = MAX(sig1 - divsigma, 0.0f);
                sig2_upd = MAX(sig2 - divsigma, 0.0f);
                
            } else if(p == INFNORM)
            {
                proj[0] = sigma * fabs(sig1);
                proj[1] = sigma * fabs(sig2);
                
                /*l1 projection part */
                sum = fLarge;
                num = 0;
                shrinkfactor = 0.0f;
                while(sum > 1.0f)
                {
                    sum = 0.0f;
                    num = 0;
                    
                    for(ii = 0; ii < 2; ii++)
                    {
                        proj[ii] = MAX(proj[ii] - shrinkfactor, 0.0f);
                        
                        sum += fabs(proj[ii]);
                        if(proj[ii]!= 0.0f)
                            num++;
                    }
                    
                    if(num > 0)
                        shrinkfactor = (sum - 1.0f) / num;
                    else
                        break;
                }
                /*l1 proj ends*/
                
                sig1_upd = sig1 - divsigma * proj[0];
                sig2_upd = sig2 - divsigma * proj[1];
            }
            
            // Compute the diagonal entries of $\widehat{\Sigma}\Sigma^{\dagger}_0$
            if(sig1 > fTiny)
                sig1_upd /= sig1;
            
            if(sig2 > fTiny)
                sig2_upd /= sig2;
            
            // Compute solution
            t1 = sig1_upd * V1 * V1 + sig2_upd * V2 * V2;
            t2 = sig1_upd * V1 * V3 + sig2_upd * V2 * V4;
            t3 = sig1_upd * V3 * V3 + sig2_upd * V4 * V4;
            
            for(k = 0; k < dimZ; k++)
            {
                gx[(dimX*dimY)*k + j*dimX + i] = vx[(dimX*dimY)*k + j*dimX + i] * t1 + vy[(dimX*dimY)*k + j*dimX + i] * t2;
                gy[(dimX*dimY)*k + j*dimX + i] = vx[(dimX*dimY)*k + j*dimX + i] * t2 + vy[(dimX*dimY)*k + j*dimX + i] * t3;
            }           
            
            // Delete allocated memory
            free(proj);
        }}
    
    return 1;
}

float divergence(float *qx_upd, float *qy_upd, float *div_upd, int dimX, int dimY, int dimZ)
{
    int i, j, k, l;
#pragma omp parallel for shared(qx_upd,qy_upd,div_upd) private(i, j, k, l)
    for(k = 0; k < dimZ; k++)   {
        for(j = 0; j < dimY; j++)   {
            l = j * dimX;            
            for(i = 0; i < dimX; i++)   {
                if(i != dimX-1)
                {
                    // ux[k][i+l] = u[k][i+1+l] - u[k][i+l]
                    div_upd[(dimX*dimY)*k + i+1+l] -= qx_upd[(dimX*dimY)*k + i+l];
                    div_upd[(dimX*dimY)*k + i+l] += qx_upd[(dimX*dimY)*k + i+l];
                }
                
                if(j != dimY-1)
                {
                    // uy[k][i+l] = u[k][i+width+l] - u[k][i+l]
                    //div_upd[(dimX*dimY)*k + i+dimY+l] -= qy_upd[(dimX*dimY)*k + i+l];
                    div_upd[(dimX*dimY)*k + i+(j+1)*dimX] -= qy_upd[(dimX*dimY)*k + i+l];                    
                    div_upd[(dimX*dimY)*k + i+l] += qy_upd[(dimX*dimY)*k + i+l];
                }
            }
        }
    }
    return *div_upd;
}

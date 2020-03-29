/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 *
 * Copyriht 2020 Suren A. Chlingaryan
 * Optimized version with 1/2 of memory consumption and ~4x performance
 * This version is algorithmicly comptable with the original, but slight change in results 
 * is expected due to different order of floating-point operations.
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

#include <malloc.h>
#include "TNV_core.h"

#define BLOCK 32
#define min(a,b) (((a)<(b))?(a):(b))

inline void coefF(float *t, float M1, float M2, float M3, float sigma, int p, int q, int r) {
    int ii, num;
    float divsigma = 1.0f / sigma;
    float sum, shrinkfactor;
    float T,D,det,eig1,eig2,sig1,sig2,V1, V2, V3, V4, v0,v1,v2, mu1,mu2,sig1_upd,sig2_upd;
    float proj[2] = {0};

    // Compute eigenvalues of M
    T = M1 + M3;
    D = M1 * M3 - M2 * M2;
    det = sqrtf(MAX((T * T / 4.0f) - D, 0.0f));
    eig1 = MAX((T / 2.0f) + det, 0.0f);
    eig2 = MAX((T / 2.0f) - det, 0.0f);
    sig1 = sqrtf(eig1);
    sig2 = sqrtf(eig2);

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
        num = 0l;
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
    t[0] = sig1_upd * V1 * V1 + sig2_upd * V2 * V2;
    t[1] = sig1_upd * V1 * V3 + sig2_upd * V2 * V4;
    t[2] = sig1_upd * V3 * V3 + sig2_upd * V4 * V4;
}


#include "hw_sched.h"
typedef struct {
    int offY, stepY, copY;
    float *Input, *u, *u_upd, *qx, *qy, *qx_upd, *qy_upd, *gradx, *grady, *gradx_upd, *grady_upd;
    float *div, *div_upd;
    float resprimal, resdual;
    float unorm, qnorm, product;
} tnv_thread_t;

typedef struct {
    int threads;
    tnv_thread_t *thr_ctx;
    float *InputT, *uT;
    int dimX, dimY, dimZ, padZ;
    float lambda, sigma, tau, theta;
} tnv_context_t;

HWSched sched = NULL;
tnv_context_t tnv_ctx;


static int tnv_free(HWThread thr, void *hwctx, int device_id, void *data) {
    int i,j,k;
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;

    free(ctx->Input);
    free(ctx->u);
    free(ctx->u_upd);
    free(ctx->qx);
    free(ctx->qy);
    free(ctx->qx_upd);
    free(ctx->qy_upd);
    free(ctx->gradx);
    free(ctx->grady);
    free(ctx->gradx_upd);
    free(ctx->grady_upd);
    free(ctx->div);
    free(ctx->div_upd);
    
    return 0;
}

static int tnv_init(HWThread thr, void *hwctx, int device_id, void *data) {
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;
    
    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int padZ = tnv_ctx->padZ;
    int offY = ctx->offY;
    int stepY = ctx->stepY;
    
//    printf("%i %p - %i %i %i x %i %i\n", device_id, ctx, dimX, dimY, dimZ, offY, stepY);

    long DimTotal = (long)(dimX*stepY*padZ);
    long Dim1Total = (long)(dimX*(stepY+1)*padZ);

    // Auxiliar vectors
    ctx->Input = memalign(64, Dim1Total * sizeof(float));
    ctx->u = memalign(64, Dim1Total * sizeof(float));
    ctx->u_upd = memalign(64, Dim1Total * sizeof(float));
    ctx->qx = memalign(64, DimTotal * sizeof(float));
    ctx->qy = memalign(64, DimTotal * sizeof(float));
    ctx->qx_upd = memalign(64, DimTotal * sizeof(float));
    ctx->qy_upd = memalign(64, DimTotal * sizeof(float));
    ctx->gradx = memalign(64, DimTotal * sizeof(float));
    ctx->grady = memalign(64, DimTotal * sizeof(float));
    ctx->gradx_upd = memalign(64, DimTotal * sizeof(float));
    ctx->grady_upd = memalign(64, DimTotal * sizeof(float));
    ctx->div = memalign(64, Dim1Total * sizeof(float));
    ctx->div_upd = malloc(Dim1Total * sizeof(float));

    if ((!ctx->Input)||(!ctx->u)||(!ctx->u_upd)||(!ctx->qx)||(!ctx->qy)||(!ctx->qx_upd)||(!ctx->qy_upd)||(!ctx->gradx)||(!ctx->grady)||(!ctx->gradx_upd)||(!ctx->grady_upd)||(!ctx->div)||(!ctx->div_upd)) {
        fprintf(stderr, "Error allocating memory\n");
        exit(-1);
    }

    return 0;
}

static int tnv_start(HWThread thr, void *hwctx, int device_id, void *data) {
    int i,j,k;
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;
    
    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int padZ = tnv_ctx->padZ;
    int offY = ctx->offY;
    int stepY = ctx->stepY;
    int copY = ctx->copY;
    
//    printf("%i %p - %i %i %i (%i) x %i %i\n", device_id, ctx, dimX, dimY, dimZ, padZ, offY, stepY);

    long DimTotal = (long)(dimX*stepY*padZ);
    long Dim1Total = (long)(dimX*copY*padZ);

    memset(ctx->u, 0, Dim1Total * sizeof(float));
    memset(ctx->qx, 0, DimTotal * sizeof(float));
    memset(ctx->qy, 0, DimTotal * sizeof(float));
    memset(ctx->gradx, 0, DimTotal * sizeof(float));
    memset(ctx->grady, 0, DimTotal * sizeof(float));
    memset(ctx->div, 0, Dim1Total * sizeof(float));
    
    for(k=0; k<dimZ; k++) {
        for(j=0; j<copY; j++) {
            for(i=0; i<dimX; i++) {
                ctx->Input[j * dimX * padZ + i * padZ + k] =  tnv_ctx->InputT[k * dimX * dimY + (j + offY) * dimX + i];
                ctx->u[j * dimX * padZ + i * padZ + k] =  tnv_ctx->uT[k * dimX * dimY + (j + offY) * dimX + i];
            }
        }
    }

    return 0;
}

static int tnv_finish(HWThread thr, void *hwctx, int device_id, void *data) {
    int i,j,k;
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;

    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int padZ = tnv_ctx->padZ;
    int offY = ctx->offY;
    int stepY = ctx->stepY;
    int copY = ctx->copY;

    for(k=0; k<dimZ; k++) {
        for(j=0; j<stepY; j++) {
            for(i=0; i<dimX; i++) {
                tnv_ctx->uT[k * dimX * dimY + (j + offY) * dimX + i] = ctx->u[j * dimX * padZ + i * padZ + k];
            }
        }
    }
    
    return 0;
}


static int tnv_copy(HWThread thr, void *hwctx, int device_id, void *data) {
    int i,j,k;
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;
    
    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int stepY = ctx->stepY;
    int copY = ctx->copY;
    int padZ = tnv_ctx->padZ;
    long DimTotal = (long)(dimX*stepY*padZ);
    long Dim1Total = (long)(dimX*copY*padZ);

    // Auxiliar vectors
    memcpy(ctx->u, ctx->u_upd, Dim1Total * sizeof(float));
    memcpy(ctx->qx, ctx->qx_upd, DimTotal * sizeof(float));
    memcpy(ctx->qy, ctx->qy_upd, DimTotal * sizeof(float));
    memcpy(ctx->gradx, ctx->gradx_upd, DimTotal * sizeof(float));
    memcpy(ctx->grady, ctx->grady_upd, DimTotal * sizeof(float));
    memcpy(ctx->div, ctx->div_upd, Dim1Total * sizeof(float));

    return 0;
}

static int tnv_restore(HWThread thr, void *hwctx, int device_id, void *data) {
    int i,j,k;
    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;
    
    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int stepY = ctx->stepY;
    int copY = ctx->copY;
    int padZ = tnv_ctx->padZ;
    long DimTotal = (long)(dimX*stepY*padZ);
    long Dim1Total = (long)(dimX*copY*padZ);

    // Auxiliar vectors
    memcpy(ctx->u_upd, ctx->u, Dim1Total * sizeof(float));
    memcpy(ctx->qx_upd, ctx->qx, DimTotal * sizeof(float));
    memcpy(ctx->qy_upd, ctx->qy, DimTotal * sizeof(float));
    memcpy(ctx->gradx_upd, ctx->gradx, DimTotal * sizeof(float));
    memcpy(ctx->grady_upd, ctx->grady, DimTotal * sizeof(float));
    memcpy(ctx->div_upd, ctx->div, Dim1Total * sizeof(float));

    return 0;
}


static int tnv_step(HWThread thr, void *hwctx, int device_id, void *data) {
    long i, j, k, l;

    tnv_context_t *tnv_ctx = (tnv_context_t*)data;
    tnv_thread_t *ctx = tnv_ctx->thr_ctx + device_id;
    
    int dimX = tnv_ctx->dimX;
    int dimY = tnv_ctx->dimY;
    int dimZ = tnv_ctx->dimZ;
    int padZ = tnv_ctx->padZ;
    int offY = ctx->offY;
    int stepY = ctx->stepY;
    int copY = ctx->copY;

    float *Input = ctx->Input;
    float *u = ctx->u;
    float *u_upd = ctx->u_upd;
    float *qx = ctx->qx;
    float *qy = ctx->qy;
    float *qx_upd = ctx->qx_upd;
    float *qy_upd = ctx->qy_upd;
    float *gradx = ctx->gradx;
    float *grady = ctx->grady;
    float *gradx_upd = ctx->gradx_upd;
    float *grady_upd = ctx->grady_upd;
    float *div = ctx->div;
    float *div_upd = ctx->div_upd;

    long p = 1l;
    long q = 1l;
    long r = 0l;

    float lambda = tnv_ctx->lambda;
    float sigma = tnv_ctx->sigma;
    float tau = tnv_ctx->tau;
    float theta = tnv_ctx->theta;
    
    float taulambda = tau * lambda;
    float divtau = 1.0f / tau;
    float divsigma = 1.0f / sigma;
    float theta1 = 1.0f + theta;
    float constant = 1.0f + taulambda;

    float resprimal = 0.0;
    float resdual = 0.0;
    float product = 0.0;
    float unorm = 0.0;
    float qnorm = 0.0;

    float udiff[dimZ] __attribute__((aligned(64)));
    float qxdiff __attribute__((aligned(64)));
    float qydiff __attribute__((aligned(64)));
    float divdiff __attribute__((aligned(64)));
    float gradxdiff[dimZ] __attribute__((aligned(64)));
    float gradydiff[dimZ] __attribute__((aligned(64)));

    for(int j1 = 0; j1 < stepY; j1 += BLOCK) {
     for(int i1 = 0; i1 < dimX; i1 += BLOCK) {
      for(int j2 = 0; j2 < BLOCK; j2++) {
        j = j1 + j2;
        for(int i2 = 0; i2 < BLOCK; i2++) {
            float t[3];
            float M1 = 0.0f, M2 = 0.0f, M3 = 0.0f;

            i = i1 + i2;
            if (i == dimX) break;
            if (j == stepY) { j2 = BLOCK; break; }
            l = (j * dimX  + i) * padZ;
        
//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < dimZ; k++) {
                u_upd[l + k] = (u[l + k] + tau * div[l + k] + taulambda * Input[l + k]) / constant;
                udiff[k] = u[l + k] - u_upd[l + k];
                unorm += (udiff[k] * udiff[k]);

                gradx_upd[l + k] = (i == (dimX - 1))?0:((u[l + k + padZ] + tau * div[l + k + padZ] + taulambda * Input[l + k + padZ]) / constant - u_upd[l + k]);
                grady_upd[l + k] = (j == (copY - 1))?0:((u[l + k + dimX*padZ] + tau * div[l + k + dimX*padZ] + taulambda * Input[l + k + dimX*padZ]) / constant - u_upd[l + k]);
                gradxdiff[k] = gradx[l + k] - gradx_upd[l + k];
                gradydiff[k] = grady[l + k] - grady_upd[l + k];

                float ubarx = theta1 * gradx_upd[l + k] - theta * gradx[l + k];
                float ubary = theta1 * grady_upd[l + k] - theta * grady[l + k];
//#define TNV_NEW_STYLE                
#ifdef TNV_NEW_STYLE                
                qx_upd[l + k] = qx[l + k] + sigma * ubarx;
                qy_upd[l + k] = qy[l + k] + sigma * ubary;

                float vx = divsigma * qx_upd[l + k]; //+ ubarx
                float vy = divsigma * qy_upd[l + k]; //+ ubary
#else
                float vx = ubarx + divsigma * qx[l + k];
                float vy = ubary + divsigma * qy[l + k];
#endif

                M1 += (vx * vx); M2 += (vx * vy); M3 += (vy * vy);
            }

            coefF(t, M1, M2, M3, sigma, p, q, r);
            
//#pragma vector aligned
#pragma GCC ivdep 
            for(k = 0; k < dimZ; k++) {
#ifdef TNV_NEW_STYLE    
                float vx = divsigma * qx_upd[l + k];
                float vy = divsigma * qy_upd[l + k];

                float gx_upd = vx * t[0] + vy * t[1];
                float gy_upd = vx * t[1] + vy * t[2];

                qx_upd[l + k] -= sigma * gx_upd;
                qy_upd[l + k] -= sigma * gy_upd;
#else
                float ubarx = theta1 * gradx_upd[l + k] - theta * gradx[l + k];
                float ubary = theta1 * grady_upd[l + k] - theta * grady[l + k];
                float vx = ubarx + divsigma * qx[l + k];
                float vy = ubary + divsigma * qy[l + k];

                float gx_upd = vx * t[0] + vy * t[1];
                float gy_upd = vx * t[1] + vy * t[2];

                qx_upd[l + k] = qx[l + k] + sigma * (ubarx - gx_upd);
                qy_upd[l + k] = qy[l + k] + sigma * (ubary - gy_upd);
#endif

                float div_upd_val = 0;
                div_upd_val -= (i > 0)?qx_upd[l + k - padZ]:0;
                div_upd_val -= (j > 0)?qy_upd[l + k - dimX * padZ]:0;
                div_upd_val += (i < (dimX-1))?qx_upd[l + k]:0;
                div_upd_val += (j < (copY-1))?qy_upd[l + k]:0;
                div_upd[l + k] = div_upd_val;

                qxdiff = qx[l + k] - qx_upd[l + k];
                qydiff = qy[l + k] - qy_upd[l + k];
                qnorm += (qxdiff * qxdiff + qydiff * qydiff);

                resdual += fabs(divsigma * qxdiff - gradxdiff[k]);
                resdual += fabs(divsigma * qydiff - gradydiff[k]);
                product += (gradxdiff[k] * qxdiff + gradydiff[k] * qydiff);

                if ((offY == 0)||(j > 0)) {
                    divdiff = div[l + k] - div_upd[l + k];  // Multiple steps... How we compute without history?
                    resprimal += fabs(divtau * udiff[k] + divdiff); 
                }
            }
            
        } // i
       } // j
      } // i
    } // j
    
    
    ctx->resprimal = resprimal;
    ctx->resdual = resdual;
    ctx->product = product;
    ctx->unorm = unorm;
    ctx->qnorm = qnorm;

    return 0;
}

static void TNV_CPU_init(float *InputT, float *uT, int dimX, int dimY, int dimZ) {
    int i, off, size, err;

    if (sched) return;

    tnv_ctx.dimX = dimX;
    tnv_ctx.dimY = dimY;
    tnv_ctx.dimZ = dimZ;
        // Padding seems actually slower
    tnv_ctx.padZ = dimZ;
//    tnv_ctx.padZ = 16 * ((dimZ / 16) + ((dimZ % 16)?1:0));
    
    hw_sched_init();

    int threads = hw_sched_get_cpu_count();
    if (threads > dimY) threads = dimY/2;

    int step = dimY / threads;
    int extra = dimY % threads;

    tnv_ctx.threads = threads;
    tnv_ctx.thr_ctx = (tnv_thread_t*)calloc(threads, sizeof(tnv_thread_t));
    for (i = 0, off = 0; i < threads; i++, off += size) {
        tnv_thread_t *ctx = tnv_ctx.thr_ctx + i;
        size = step + ((i < extra)?1:0);

        ctx->offY = off;
        ctx->stepY = size;

        if (i == (threads-1)) ctx->copY = ctx->stepY;
        else ctx->copY = ctx->stepY + 1;
    }

    sched = hw_sched_create(threads);
    if (!sched) { fprintf(stderr, "Error creating threads\n"); exit(-1); }

    err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_init);
    if (!err) err = hw_sched_wait_task(sched);
    if (err) { fprintf(stderr, "Error %i scheduling init threads", err); exit(-1); }
}



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
 * 1. Filtered/regularized image (u)
 *
 * [1]. Duran, J., Moeller, M., Sbert, C. and Cremers, D., 2016. Collaborative total variation: a general framework for vectorial TV models. SIAM Journal on Imaging Sciences, 9(1), pp.116-151.
 */

float TNV_CPU_main(float *InputT, float *uT, float lambda, int maxIter, float tol, int dimX, int dimY, int dimZ)
{
    int err;
    int iter;
    int i,j,k,l,m;

    lambda = 1.0f/(2.0f*lambda);
    tnv_ctx.lambda = lambda;

    // PDHG algorithm parameters
    float tau = 0.5f;
    float sigma = 0.5f;
    float theta = 1.0f;

    // Backtracking parameters
    float s = 1.0f;
    float gamma = 0.75f;
    float beta = 0.95f;
    float alpha0 = 0.2f;
    float alpha = alpha0;
    float delta = 1.5f;
    float eta = 0.95f;

    TNV_CPU_init(InputT, uT, dimX, dimY, dimZ);

    tnv_ctx.InputT = InputT;
    tnv_ctx.uT = uT;
    
    int padZ = tnv_ctx.padZ;

    err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_start);
    if (!err) err = hw_sched_wait_task(sched);
    if (err) { fprintf(stderr, "Error %i scheduling start threads", err); exit(-1); }


    // Apply Primal-Dual Hybrid Gradient scheme
    float residual = fLarge;
    for(iter = 0; iter < maxIter; iter++)   {
        float resprimal = 0.0f;
        float resdual = 0.0f;
        float product = 0.0f;
        float unorm = 0.0f;
        float qnorm = 0.0f;

        float divtau = 1.0f / tau;

        tnv_ctx.sigma = sigma;
        tnv_ctx.tau = tau;
        tnv_ctx.theta = theta;

        err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_step);
        if (!err) err = hw_sched_wait_task(sched);
        if (err) { fprintf(stderr, "Error %i scheduling tnv threads", err); exit(-1); }

            // border regions
        for (j = 1; j < tnv_ctx.threads; j++) {
            tnv_thread_t *ctx0 = tnv_ctx.thr_ctx + (j - 1);
            tnv_thread_t *ctx = tnv_ctx.thr_ctx + j;

            m = ctx0->stepY  * dimX * padZ;
            for(i = 0; i < dimX; i++) {
                for(k = 0; k < dimZ; k++) {
                    int l = i * padZ + k;
                        
                    ctx->div_upd[l] -= ctx0->qy_upd[m - dimX * padZ + l];
                    ctx0->div_upd[m + l] = ctx->div_upd[l];
                    ctx0->u_upd[m + l] = ctx->u_upd[l];

                    float divdiff = ctx->div[l] - ctx->div_upd[l];
                    float udiff = ctx->u[l] - ctx->u_upd[l];
                    resprimal += fabs(divtau * udiff + divdiff); 
                }
            }
        }

        for (j = 0; j < tnv_ctx.threads; j++) {
            tnv_thread_t *ctx = tnv_ctx.thr_ctx + j;
            resprimal += ctx->resprimal;
            resdual += ctx->resdual;
            product += ctx->product;
            unorm += ctx->unorm;
            qnorm += ctx->qnorm;
        } 

        residual = (resprimal + resdual) / ((float) (dimX*dimY*dimZ));
        float b = (2.0f * tau * sigma * product) / (gamma * sigma * unorm + gamma * tau * qnorm);
        float dual_dot_delta = resdual * s * delta;
        float dual_div_delta = (resdual * s) / delta;
        printf("resprimal: %f, resdual: %f, b: %f (product: %f, unorm: %f, qnorm: %f)\n", resprimal, resdual, b, product, unorm, qnorm);


        if(b > 1) {
            // Decrease step-sizes to fit balancing principle
            tau = (beta * tau) / b;
            sigma = (beta * sigma) / b;
            alpha = alpha0;

            err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_restore);
            if (!err) err = hw_sched_wait_task(sched);
            if (err) { fprintf(stderr, "Error %i scheduling restore threads", err); exit(-1); }
        } else {
            err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_copy);
            if (!err) err = hw_sched_wait_task(sched);
            if (err) { fprintf(stderr, "Error %i scheduling copy threads", err); exit(-1); }
        
            if(resprimal > dual_dot_delta) {
                    // Increase primal step-size and decrease dual step-size
                tau = tau / (1.0f - alpha);
                sigma = sigma * (1.0f - alpha);
                alpha = alpha * eta;
            } else if(resprimal < dual_div_delta) {
                    // Decrease primal step-size and increase dual step-size
                tau = tau * (1.0f - alpha);
                sigma = sigma / (1.0f - alpha);
                alpha = alpha * eta;
            }
        }

        if (residual < tol) break;
    }

    err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_finish);
    if (!err) err = hw_sched_wait_task(sched);
    if (err) { fprintf(stderr, "Error %i scheduling finish threads", err); exit(-1); }


    printf("Iterations stopped at %i with the residual %f \n", iter, residual);
    printf("Return: %f\n", *uT);

    return *uT;
}

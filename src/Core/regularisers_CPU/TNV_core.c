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
    float *Input, *u, *qx, *qy, *gradx, *grady, *div;
    float *div0, *udiff0;
    float *gradxdiff, *gradydiff, *ubarx, *ubary, *udiff;
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
    free(ctx->qx);
    free(ctx->qy);
    free(ctx->gradx);
    free(ctx->grady);
    free(ctx->div);
    
    free(ctx->div0);
    free(ctx->udiff0);

    free(ctx->gradxdiff); 
    free(ctx->gradydiff);
    free(ctx->ubarx);
    free(ctx->ubary);
    free(ctx->udiff);

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
    long DimRow = (long)(dimX * padZ);
    long DimCell = (long)(padZ);

    // Auxiliar vectors
    ctx->Input = memalign(64, Dim1Total * sizeof(float));
    ctx->u = memalign(64, Dim1Total * sizeof(float));
    ctx->qx = memalign(64, DimTotal * sizeof(float));
    ctx->qy = memalign(64, DimTotal * sizeof(float));
    ctx->gradx = memalign(64, DimTotal * sizeof(float));
    ctx->grady = memalign(64, DimTotal * sizeof(float));
    ctx->div = memalign(64, Dim1Total * sizeof(float));

    ctx->div0 = memalign(64, DimRow * sizeof(float));
    ctx->udiff0 = memalign(64, DimRow * sizeof(float));

    ctx->gradxdiff = memalign(64, DimCell * sizeof(float));
    ctx->gradydiff = memalign(64, DimCell * sizeof(float));
    ctx->ubarx = memalign(64, DimCell * sizeof(float));
    ctx->ubary = memalign(64, DimCell * sizeof(float));
    ctx->udiff = memalign(64, DimCell * sizeof(float));
    
    if ((!ctx->Input)||(!ctx->u)||(!ctx->qx)||(!ctx->qy)||(!ctx->gradx)||(!ctx->grady)||(!ctx->div)||(!ctx->div0)||(!ctx->udiff)||(!ctx->udiff0)) {
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

    memset(ctx->u, 0, Dim1Total * sizeof(float));
    memset(ctx->qx, 0, DimTotal * sizeof(float));
    memset(ctx->qy, 0, DimTotal * sizeof(float));
    memset(ctx->gradx, 0, DimTotal * sizeof(float));
    memset(ctx->grady, 0, DimTotal * sizeof(float));
    memset(ctx->div, 0, Dim1Total * sizeof(float));

    return 0;
}


static int tnv_step(HWThread thr, void *hwctx, int device_id, void *data) {
    long i, j, k, l, m;

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
    float *qx = ctx->qx;
    float *qy = ctx->qy;
    float *gradx = ctx->gradx;
    float *grady = ctx->grady;
    float *div = ctx->div;

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

    float resprimal = 0.0f;
    float resdual1 = 0.0f;
    float resdual2 = 0.0f;
    float product = 0.0f;
    float unorm = 0.0f;
    float qnorm = 0.0f;

    float qxdiff;
    float qydiff;
    float divdiff;
    float *gradxdiff = ctx->gradxdiff;
    float *gradydiff = ctx->gradydiff;
    float *ubarx = ctx->ubarx;
    float *ubary = ctx->ubary;
    float *udiff = ctx->udiff;

    float *udiff0 = ctx->udiff0;
    float *div0 = ctx->div0;


    j = 0; {
#       define TNV_LOOP_FIRST_J
        i = 0; {
#           define TNV_LOOP_FIRST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_FIRST_I
        }
        for(i = 1; i < (dimX - 1); i++) {
#           include "TNV_core_loop.h"
        }
        i = dimX - 1; {
#           define TNV_LOOP_LAST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_LAST_I
        }
#       undef TNV_LOOP_FIRST_J
    }



    for(int j = 1; j < (copY - 1); j++) {
        i = 0; {
#           define TNV_LOOP_FIRST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_FIRST_I
        }
    }

    for(int j1 = 1; j1 < (copY - 1); j1 += BLOCK) {
        for(int i1 = 1; i1 < (dimX - 1); i1 += BLOCK) {
            for(int j2 = 0; j2 < BLOCK; j2 ++) {
                j = j1 + j2;
                for(int i2 = 0; i2 < BLOCK; i2++) {
                    i = i1 + i2;
                    
                    if (i == (dimX - 1)) break;
                    if (j == (copY - 1)) { j2 = BLOCK; break; }
#           include "TNV_core_loop.h"
                }   
            }
        } // i

    }

    for(int j = 1; j < (copY - 1); j++) {
        i = dimX - 1; {
#           define TNV_LOOP_LAST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_LAST_I
        }
    }



    for (j = copY - 1; j < stepY; j++) {
#       define TNV_LOOP_LAST_J
        i = 0; {
#           define TNV_LOOP_FIRST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_FIRST_I
        }
        for(i = 1; i < (dimX - 1); i++) {
#           include "TNV_core_loop.h"
        }
        i = dimX - 1; {
#           define TNV_LOOP_LAST_I
#           include "TNV_core_loop.h"
#           undef TNV_LOOP_LAST_I
        }
#       undef TNV_LOOP_LAST_J
    }



    ctx->resprimal = resprimal;
    ctx->resdual = resdual1 + resdual2;
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
//    tnv_ctx.padZ = dimZ;
//    tnv_ctx.padZ = 4 * ((dimZ / 4) + ((dimZ % 4)?1:0));
    tnv_ctx.padZ = 16 * ((dimZ / 16) + ((dimZ % 16)?1:0));
    
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
    int started = 0;
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

            m = (ctx0->stepY - 1) * dimX * padZ;
            for(i = 0; i < dimX; i++) {
                for(k = 0; k < dimZ; k++) {
                    int l = i * padZ + k;
                        
                    float divdiff = ctx->div0[l] - ctx->div[l];
                    float udiff = ctx->udiff0[l];

                    ctx->div[l] -= ctx0->qy[l + m];
                    ctx0->div[m + l + dimX*padZ] = ctx->div[l];
                    ctx0->u[m + l + dimX*padZ] = ctx->u[l];
                    
                    divdiff += ctx0->qy[l + m];
                    resprimal += fabs(divtau * udiff + divdiff); 
                }
            }
        }

        {
            tnv_thread_t *ctx = tnv_ctx.thr_ctx + 0;
            for(i = 0; i < dimX; i++) {
                for(k = 0; k < dimZ; k++) {
                    int l = i * padZ + k;
                        
                    float divdiff = ctx->div0[l] - ctx->div[l];
                    float udiff = ctx->udiff0[l];
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
//        printf("resprimal: %f, resdual: %f, b: %f (product: %f, unorm: %f, qnorm: %f)\n", resprimal, resdual, b, product, unorm, qnorm);


        if(b > 1) {
            
            // Decrease step-sizes to fit balancing principle
            tau = (beta * tau) / b;
            sigma = (beta * sigma) / b;
            alpha = alpha0;

            if (started) {
                fprintf(stderr, "\n\n\nWARNING: Back-tracking is required in the middle of iterative optimization! We CAN'T do it in the fast version. The standard TNV recommended\n\n\n");
            } else {
                err = hw_sched_schedule_thread_task(sched, (void*)&tnv_ctx, tnv_restore);
                if (!err) err = hw_sched_wait_task(sched);
                if (err) { fprintf(stderr, "Error %i scheduling restore threads", err); exit(-1); }
            }
        } else {
            started = 1;
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
//    printf("Return: %f\n", *uT);

    return *uT;
}

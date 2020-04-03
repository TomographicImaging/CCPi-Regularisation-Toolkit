/*
 * The PyHST program is Copyright (C) 2002-2011 of the
 * European Synchrotron Radiation Facility (ESRF) and
 * Karlsruhe Institute of Technology (KIT).
 *
 * PyHST is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * hst is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hw_sched.h"
#include "hw_thread.h"

#ifdef HW_USE_THREADS
static void *hw_thread_function(HWThread ctx) {
    int err;
    int chunk_id;
    
    HWRunFunction *runs;
    HWRunFunction run;
    HWSched sched;
    void *hwctx;
    
    sched = ctx->sched;
    runs = ctx->run;
    hwctx = ctx->hwctx;
    
    hw_sched_lock(sched, job_cond);

    hw_sched_lock(sched, compl_cond);
    ctx->status = HW_THREAD_STATUS_IDLE;    
    hw_sched_broadcast(sched, compl);
    hw_sched_unlock(sched, compl_cond);
    
    while (sched->status) {
	hw_sched_wait(sched, job);
	if (!sched->status) break;

	ctx->err = 0;
	ctx->status = HW_THREAD_STATUS_STARTING;
	hw_sched_unlock(sched, job_cond);
	
	run = hw_run_entry(runs, sched->entry);
#if 0
	// Offset to interleave transfers if the GPUBox is used
	// Just check with CUDA_LAUNCH_BLOCKED the togpu time and put it here
	// It should be still significantly less than BP time
	// We can do callibration during initilization in future
	
	usleep(12000 * ctx->thread_id);
#endif
	chunk_id = hw_sched_get_chunk(sched, ctx->thread_id);

	    /* Should be after get_chunk, since we can check if it's first time */
	ctx->status = HW_THREAD_STATUS_RUNNING; 
	while (chunk_id != HW_SCHED_CHUNK_INVALID) {
	    //printf("Thread %i processing slice %i\n", ctx->thread_id, chunk_id);
	    err = run(ctx, hwctx, chunk_id, sched->ctx);
	    if (err) {
		ctx->err = err;
		break;
	    }
	    chunk_id = hw_sched_get_chunk(sched, ctx->thread_id);
	}

	hw_sched_lock(sched, job_cond);
	
	hw_sched_lock(sched, compl_cond);
	ctx->status = HW_THREAD_STATUS_DONE;
	hw_sched_broadcast(sched, compl);
	hw_sched_unlock(sched, compl_cond);
    }

    hw_sched_unlock(sched, job_cond);

    g_thread_exit(NULL);
    return NULL; /* TODO: check this */
}
#endif /* HW_USE_THREADS */


HWThread hw_thread_create(HWSched sched, int thread_id, void *hwctx, HWRunFunction *run_func, HWFreeFunction free_func) {
    GError *err;
    
    HWThread ctx;
    
    ctx = (HWThread)malloc(sizeof(HWThreadS));
    if (!ctx) return ctx;
    
    memset(ctx, 0, sizeof(HWThreadS));

    ctx->sched = sched;
    ctx->hwctx = hwctx;
    ctx->run = run_func;
    ctx->free = free_func;
    ctx->thread_id = thread_id;
    ctx->status = HW_THREAD_STATUS_INIT;
    
#ifdef HW_USE_THREADS
    ctx->thread = g_thread_create((GThreadFunc)hw_thread_function, ctx, 1, &err);
    if (!ctx->thread) {
	g_error_free(err);

	hw_thread_destroy(ctx);
	return NULL;
    }
#endif /* HW_USE_THREADS */
    
    return ctx;
}

void hw_thread_destroy(HWThread ctx) {
#ifdef HW_USE_THREADS
    if (ctx->thread) {
	g_thread_join(ctx->thread);
    }
#endif /* HW_USE_THREADS */
    
    if (ctx->data) {
	free(ctx->data);
    }
    
    if (ctx->free) {
	ctx->free(ctx->hwctx);
    }
    
    free(ctx);
}

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

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HW_HAVE_SCHED_HEADERS
# include <sys/types.h>
# include <unistd.h>
# include <sched.h>
#endif /* HW_HAVE_SCHED_HEADERS */

#include "hw_sched.h"


#ifdef HW_USE_THREADS
# define MUTEX_INIT(ctx, name) \
    if (!err) { \
	ctx->name##_mutex = g_mutex_new(); \
	if (!ctx->name##_mutex) err = 1; \
    }
    
# define MUTEX_FREE(ctx, name) \
    if (ctx->name##_mutex) g_mutex_free(ctx->name##_mutex);

# define COND_INIT(ctx, name) \
    MUTEX_INIT(ctx, name##_cond) \
    if (!err) { \
	ctx->name##_cond = g_cond_new(); \
	if (!ctx->name##_cond) { \
	    err = 1; \
	    MUTEX_FREE(ctx, name##_cond) \
	} \
    }

# define COND_FREE(ctx, name) \
    if (ctx->name##_cond) g_cond_free(ctx->name##_cond); \
    MUTEX_FREE(ctx, name##_cond)
#else /* HW_USE_THREADS */
# define MUTEX_INIT(ctx, name)
# define MUTEX_FREE(ctx, name)
# define COND_INIT(ctx, name)
# define COND_FREE(ctx, name)
#endif /* HW_USE_THREADS */


HWRunFunction ppu_run[] = {
    (HWRunFunction)NULL
};

static int hw_sched_initialized = 0;

int hw_sched_init(void) {
    if (!hw_sched_initialized) {
#ifdef HW_USE_THREADS
	g_thread_init(NULL);
#endif /* HW_USE_THREADS */
	hw_sched_initialized = 1;
    }

    return 0;
}


int hw_sched_get_cpu_count(void) {
#ifdef HW_HAVE_SCHED_HEADERS
    int err;

    int cpu_count;
    cpu_set_t mask;

    err = sched_getaffinity(getpid(), sizeof(mask), &mask);
    if (err) return 1;

# ifdef CPU_COUNT
    cpu_count = CPU_COUNT(&mask);
# else
    for (cpu_count = 0; cpu_count < CPU_SETSIZE; cpu_count++) {
	if (!CPU_ISSET(cpu_count, &mask)) break;
    }
# endif

    if (!cpu_count) cpu_count = 1;
    return cpu_count;    
#else /* HW_HAVE_SCHED_HEADERS */
    return 1;
#endif /* HW_HAVE_SCHED_HEADERS */
}


HWSched hw_sched_create(int cpu_count) {
    int i;
    int err = 0;

    HWSched ctx;

    //hw_sched_init();
    
    ctx = (HWSched)malloc(sizeof(HWSchedS));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(HWSchedS));

    ctx->status = 1;

    MUTEX_INIT(ctx, sync);
    MUTEX_INIT(ctx, data);
    COND_INIT(ctx, compl);
    COND_INIT(ctx, job);
    
    if (err) {
	hw_sched_destroy(ctx);
	return NULL;
    }
    
    if (!cpu_count) cpu_count = hw_sched_get_cpu_count();
    if (cpu_count > HW_MAX_THREADS) cpu_count = HW_MAX_THREADS;

    ctx->n_threads = 0;
    for (i = 0; i < cpu_count; i++) {
	ctx->thread[ctx->n_threads] = hw_thread_create(ctx, ctx->n_threads, NULL, ppu_run, NULL);
	if (ctx->thread[ctx->n_threads]) {
#ifndef HW_USE_THREADS
	    ctx->thread[ctx->n_threads]->status = HW_THREAD_STATUS_STARTING;
#endif /* HW_USE_THREADS */
	    ++ctx->n_threads;
	}
    }
    
    if (!ctx->n_threads) {
	hw_sched_destroy(ctx);
	return NULL;
    }
    
    return ctx;
}

static int hw_sched_wait_threads(HWSched ctx) {
#ifdef HW_USE_THREADS
    int i = 0;
    
    hw_sched_lock(ctx, compl_cond);
    while (i < ctx->n_threads) {
        for (; i < ctx->n_threads; i++) {
	    if (ctx->thread[i]->status == HW_THREAD_STATUS_INIT) {
		hw_sched_wait(ctx, compl);
		break;
	    }
	}
	
    }
    hw_sched_unlock(ctx, compl_cond);
#endif /* HW_USE_THREADS */
    
    ctx->started = 1;

    return 0;
}

void hw_sched_destroy(HWSched ctx) {
    int i;

    if (ctx->n_threads > 0) {
	if (!ctx->started) {
	    hw_sched_wait_threads(ctx);
	}

	ctx->status = 0;
	hw_sched_lock(ctx, job_cond);
	hw_sched_broadcast(ctx, job);
	hw_sched_unlock(ctx, job_cond);
    
	for (i = 0; i < ctx->n_threads; i++) {
	    hw_thread_destroy(ctx->thread[i]);
	}
    }

    COND_FREE(ctx, job);
    COND_FREE(ctx, compl);
    MUTEX_FREE(ctx, data);
    MUTEX_FREE(ctx, sync);

    free(ctx);
}

int hw_sched_set_sequential_mode(HWSched ctx, int *n_blocks, int *cur_block, HWSchedFlags flags) {
    ctx->mode = HW_SCHED_MODE_SEQUENTIAL;
    ctx->n_blocks = n_blocks;
    ctx->cur_block = cur_block;
    ctx->flags = flags;
    
    return 0;
}

int hw_sched_get_chunk(HWSched ctx, int thread_id) {
    int block;

    switch (ctx->mode) {
	case HW_SCHED_MODE_PREALLOCATED:
	    if (ctx->thread[thread_id]->status == HW_THREAD_STATUS_STARTING) {
#ifndef HW_USE_THREADS
	        ctx->thread[thread_id]->status = HW_THREAD_STATUS_DONE;
#endif /* HW_USE_THREADS */
                return thread_id;
	    } else {
		return HW_SCHED_CHUNK_INVALID;
	    }
	case HW_SCHED_MODE_SEQUENTIAL:
	    if ((ctx->flags&HW_SCHED_FLAG_INIT_CALL)&&(ctx->thread[thread_id]->status == HW_THREAD_STATUS_STARTING)) {
	        return HW_SCHED_CHUNK_INIT;
	    }
	    hw_sched_lock(ctx, data);
	    block = *ctx->cur_block;
	    if (block < *ctx->n_blocks) {
		*ctx->cur_block = *ctx->cur_block + 1;
	    } else {
		block = HW_SCHED_CHUNK_INVALID;
	    }
	    hw_sched_unlock(ctx, data);
	    if (block == HW_SCHED_CHUNK_INVALID) {
	        if (((ctx->flags&HW_SCHED_FLAG_FREE_CALL)&&(ctx->thread[thread_id]->status == HW_THREAD_STATUS_RUNNING))) {
	            ctx->thread[thread_id]->status = HW_THREAD_STATUS_FINISHING;
	            return HW_SCHED_CHUNK_FREE;
	        }
	        if ((ctx->flags&HW_SCHED_FLAG_TERMINATOR_CALL)&&((ctx->thread[thread_id]->status == HW_THREAD_STATUS_RUNNING)||(ctx->thread[thread_id]->status == HW_THREAD_STATUS_FINISHING))) {
	            int i;
	            hw_sched_lock(ctx, data);
	            for (i = 0; i < ctx->n_threads; i++) {
	                if (thread_id == i) continue;
	                if ((ctx->thread[i]->status != HW_THREAD_STATUS_DONE)&&(ctx->thread[i]->status != HW_THREAD_STATUS_FINISHING2)&&(ctx->thread[i]->status != HW_THREAD_STATUS_IDLE)) {
	            	    break;
	            	}
	            }
	            ctx->thread[thread_id]->status  = HW_THREAD_STATUS_FINISHING2;
	            hw_sched_unlock(ctx, data);
	            if (i == ctx->n_threads) {
	                return HW_SCHED_CHUNK_TERMINATOR;
	            } 
	        }
	    }
	    return block;
	default:
	    return HW_SCHED_CHUNK_INVALID;
    }

    return -1;
}

    
int hw_sched_schedule_task(HWSched ctx, void *appctx, HWEntry entry) {
#ifdef HW_USE_THREADS
    if (!ctx->started) {
	hw_sched_wait_threads(ctx);
    }
#else /* HW_USE_THREADS */
    int err;
    int i, chunk_id, n_threads;
    HWRunFunction run;
    HWThread thrctx;
#endif /* HW_USE_THREADS */
    
    ctx->ctx = appctx;
    ctx->entry = entry;

    switch (ctx->mode) {
	case HW_SCHED_MODE_SEQUENTIAL:
	    *ctx->cur_block = 0;
	break;
	default:
	    ;
    }
        
#ifdef HW_USE_THREADS
    hw_sched_lock(ctx, compl_cond);

    hw_sched_lock(ctx, job_cond);
    hw_sched_broadcast(ctx, job);
    hw_sched_unlock(ctx, job_cond);
#else  /* HW_USE_THREADS */
    n_threads = ctx->n_threads;
    
    for (i = 0; i < n_threads; i++) {
	thrctx = ctx->thread[i];
	thrctx->err = 0;
    }

    i = 0;
    thrctx = ctx->thread[i];
    chunk_id = hw_sched_get_chunk(ctx, thrctx->thread_id);

    while (chunk_id >= 0) {
	run = hw_run_entry(thrctx->runs, entry);
        err = run(thrctx, thrctx->hwctx, chunk_id, appctx);
	if (err) {
	    thrctx->err = err;
	    break;
	}
	
	if ((++i) == n_threads) i = 0;
	thrctx = ctx->thread[i];
        chunk_id = hw_sched_get_chunk(ctx, thrctx->thread_id);
    }
#endif /* HW_USE_THREADS */

    return 0;
}

int hw_sched_wait_task(HWSched ctx) {
    int err = 0;
    int i = 0, n_threads = ctx->n_threads;

#ifdef HW_USE_THREADS
    while (i < ctx->n_threads) {
        for (; i < ctx->n_threads; i++) {
	    if (ctx->thread[i]->status == HW_THREAD_STATUS_DONE) {
		ctx->thread[i]->status = HW_THREAD_STATUS_IDLE;
	    } else {
		hw_sched_wait(ctx, compl);
		break;
	    }
	}
	
    }

    hw_sched_unlock(ctx, compl_cond);
#endif /* HW_USE_THREADS */

    for (i = 0; i < n_threads; i++) {
	HWThread thrctx = ctx->thread[i];
	if (thrctx->err) return err = thrctx->err;

#ifndef HW_USE_THREADS
        thrctx->status = HW_THREAD_STATUS_IDLE;
#endif /* HW_USE_THREADS */
    }

    return err;
}

int hw_sched_execute_task(HWSched ctx, void *appctx, HWEntry entry) {
    int err;
    
    err = hw_sched_schedule_task(ctx, appctx, entry);
    if (err) return err;
    
    return hw_sched_wait_task(ctx);
}

int hw_sched_schedule_thread_task(HWSched ctx, void *appctx, HWEntry entry) {
    int err;
    
    ctx->saved_mode = ctx->mode;
    ctx->mode = HW_SCHED_MODE_PREALLOCATED;
    err = hw_sched_schedule_task(ctx, appctx, entry);
    
    return err;
}


int hw_sched_wait_thread_task(HWSched ctx) {
    int err;

    err = hw_sched_wait_task(ctx);
    ctx->mode = ctx->saved_mode;

    return err;
}

int hw_sched_execute_thread_task(HWSched ctx, void *appctx, HWEntry entry) {
    int err;
    int saved_mode = ctx->mode;

    ctx->mode = HW_SCHED_MODE_PREALLOCATED;
    err = hw_sched_execute_task(ctx, appctx, entry);
    ctx->mode = saved_mode;
    
    return err;
}

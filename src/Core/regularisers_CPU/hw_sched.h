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

#ifndef _HW_SCHED_H
#define _HW_SCHED_H

#include <glib.h>

    // enable threading
#define HW_HAVE_SCHED_HEADERS
#define HW_USE_THREADS


typedef struct HWSchedT *HWSched;
#ifdef HW_USE_THREADS
typedef GMutex *HWMutex;
#else /* HW_USE_THREADS */
typedef void *HWMutex;
#endif /* HW_USE_THREADS */


#include "hw_thread.h"

enum HWSchedModeT {
    HW_SCHED_MODE_PREALLOCATED = 0,
    HW_SCHED_MODE_SEQUENTIAL
};
typedef enum HWSchedModeT HWSchedMode;

enum HWSchedChunkT {
    HW_SCHED_CHUNK_INVALID = -1,
    HW_SCHED_CHUNK_INIT = -2,
    HW_SCHED_CHUNK_FREE = -3,
    HW_SCHED_CHUNK_TERMINATOR = -4
};
typedef enum HWSchedChunkT HWSchedChunk;

enum HWSchedFlagsT {
    HW_SCHED_FLAG_INIT_CALL = 1,        //! Executed in each thread before real chunks
    HW_SCHED_FLAG_FREE_CALL = 2,        //! Executed in each thread after real chunks
    HW_SCHED_FLAG_TERMINATOR_CALL = 4   //! Executed in one of the threads after all threads are done
};
typedef enum HWSchedFlagsT HWSchedFlags;


#define HW_SINGLE_MODE
//#define HW_DETECT_CPU_CORES
#define HW_MAX_THREADS 128

#ifdef HW_SINGLE_MODE
    typedef HWRunFunction HWEntry;
# define hw_run_entry(runs, entry) entry
#else /* HW_SINGLE_MODE */
    typedef int HWEntry;
# define hw_run_entry(runs, entry) runs[entry]
#endif /* HW_SINGLE_MODE */

#ifndef HW_HIDE_DETAILS
struct HWSchedT {
    int status;
    int started;
    
    int n_threads;
    HWThread thread[HW_MAX_THREADS];
    
#ifdef HW_USE_THREADS
    GCond *job_cond, *compl_cond;
    GMutex *job_cond_mutex, *compl_cond_mutex, *data_mutex;
    GMutex *sync_mutex;
#endif /* HW_USE_THREADS */
    
    HWSchedMode mode;
    HWSchedMode saved_mode;
    HWSchedFlags flags;
    int *n_blocks;
    int *cur_block;

    HWEntry entry;
    void *ctx;
};
typedef struct HWSchedT HWSchedS;
#endif /* HW_HIDE_DETAILS */

# ifdef __cplusplus
extern "C" {
# endif

HWSched hw_sched_create(int ppu_count);
int hw_sched_init(void);
void hw_sched_destroy(HWSched ctx);
int hw_sched_get_cpu_count(void);

int hw_sched_set_sequential_mode(HWSched ctx, int *n_blocks, int *cur_block, HWSchedFlags flags);
int hw_sched_get_chunk(HWSched ctx, int thread_id);
int hw_sched_schedule_task(HWSched ctx, void *appctx, HWEntry entry);
int hw_sched_wait_task(HWSched ctx);
int hw_sched_execute_task(HWSched ctx, void *appctx, HWEntry entry);

int hw_sched_schedule_thread_task(HWSched ctx, void *appctx, HWEntry entry);
int hw_sched_wait_thread_task(HWSched ctx);
int hw_sched_execute_thread_task(HWSched ctx, void *appctx, HWEntry entry);

HWMutex hw_sched_create_mutex(void);
void hw_sched_destroy_mutex(HWMutex ctx);

#ifdef HW_USE_THREADS
# define hw_sched_lock(ctx, type) g_mutex_lock(ctx->type##_mutex)
# define hw_sched_unlock(ctx, type) g_mutex_unlock(ctx->type##_mutex)
# define hw_sched_broadcast(ctx, type) g_cond_broadcast(ctx->type##_cond)
# define hw_sched_signal(ctx, type) g_cond_signal(ctx->type##_cond)
# define hw_sched_wait(ctx, type) g_cond_wait(ctx->type##_cond, ctx->type##_cond_mutex)

#define hw_sched_create_mutex(void) g_mutex_new()
#define hw_sched_destroy_mutex(ctx) g_mutex_free(ctx)
#define hw_sched_lock_mutex(ctx) g_mutex_lock(ctx)
#define hw_sched_unlock_mutex(ctx) g_mutex_unlock(ctx)
#else /* HW_USE_THREADS */
# define hw_sched_lock(ctx, type)
# define hw_sched_unlock(ctx, type)
# define hw_sched_broadcast(ctx, type)
# define hw_sched_signal(ctx, type)
# define hw_sched_wait(ctx, type)

#define hw_sched_create_mutex(void) NULL
#define hw_sched_destroy_mutex(ctx)
#define hw_sched_lock_mutex(ctx)
#define hw_sched_unlock_mutex(ctx)
#endif /* HW_USE_THREADS */

# ifdef __cplusplus
}
# endif

#endif /* _HW_SCHED_H */


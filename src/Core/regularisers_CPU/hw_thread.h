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

#ifndef _HW_THREAD_H
#define _HW_THREAD_H

#include <glib.h>

typedef struct HWThreadT *HWThread;
typedef int (*HWRunFunction)(HWThread thread, void *ctx, int block, void *attr);
typedef int (*HWFreeFunction)(void *ctx);

#include "hw_sched.h"

enum HWThreadStatusT {
    HW_THREAD_STATUS_IDLE = 0,
    HW_THREAD_STATUS_STARTING = 1,
    HW_THREAD_STATUS_RUNNING = 2,
    HW_THREAD_STATUS_FINISHING = 3,
    HW_THREAD_STATUS_FINISHING2 = 4,
    HW_THREAD_STATUS_DONE = 5,
    HW_THREAD_STATUS_INIT = 6
};
typedef enum HWThreadStatusT HWThreadStatus;


#ifndef HW_HIDE_DETAILS
struct HWThreadT {
    int thread_id;
    HWSched sched;

#ifdef HW_USE_THREADS    
    GThread *thread;
#endif /* HW_USE_THREADS */
    
    void *hwctx;
    HWRunFunction *run;
    HWFreeFunction free;
    
    int err;
    HWThreadStatus status;

    void *data;			/**< Per-thread data storage, will be free'd if set */
};
typedef struct HWThreadT HWThreadS;
#endif /* HW_HIDE_DETAILS */

# ifdef __cplusplus
extern "C" {
# endif

HWThread hw_thread_create(HWSched sched, int thread_id, void *hwctx, HWRunFunction *run_func, HWFreeFunction free_func);
void hw_thread_destroy(HWThread ctx);

# ifdef __cplusplus
}
# endif


#endif /* _HW_THREAD_H */

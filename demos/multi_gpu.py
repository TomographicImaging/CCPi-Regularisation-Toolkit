#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A script to demonstrate multi-gpu capabilities of the regularisation package, note
that mpi4py is required for this to work.

#############################################################################
Run the demo for two processes for e.g.:
mpirun -np 2 python multi_gpu.py -g -s -gpus 1
#############################################################################

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev
"""


def data_generator():
    import numpy as np

    vol_size = 160
    vol3d = np.float32(np.random.rand(vol_size, vol_size, vol_size))
    return vol3d


def filter3D(vol3d, iterations_reg, DEVICE_no):
    from ccpi.filters.regularisers import ROF_TV

    # perform basic data splitting between GPUs
    print("-----------------------------------------------------------------")
    print("Perform 3D filtering on {} GPU device...".format(DEVICE_no))
    print("-----------------------------------------------------------------")
    # set parameters
    pars = {
        "algorithm": ROF_TV,
        "input": vol3d,
        "regularisation_parameter": 0.01,
        "number_of_iterations": iterations_reg,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }

    (rof_gpu3D, info_vec_gpu) = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        DEVICE_no,
    )

    return rof_gpu3D


# %%
if __name__ == "__main__":
    # imports
    from mpi4py import MPI

    # MPI process
    mpi_proc_num = MPI.COMM_WORLD.size
    mpi_proc_id = MPI.COMM_WORLD.rank

    # process arguments
    import argparse

    parser = argparse.ArgumentParser(description="GPU device use from mpi4py")
    parser.add_argument(
        "-g",
        "--get_device",
        action="store_true",
        help="report device for each MPI process (default: NO)",
    )
    parser.add_argument(
        "-s",
        "--set_device",
        action="store_true",
        help="automatically set device for each MPI process (default: NO)",
    )
    parser.add_argument(
        "-gpus",
        "--gpus_no",
        dest="gpus_total",
        default=2,
        help="the total number of available GPU devices",
    )
    args = parser.parse_args()

    # Generating the projection data
    # NOTE that the data is generated for each mpi process for the sake of simplicity but it could be splitted
    # into multiple mpi processess generating smaller chunks of the global dataset
    vol3d = data_generator()

    # set the total number of available GPU devices
    GPUs_total_num = int(args.gpus_total)
    DEVICE_no = mpi_proc_id % GPUs_total_num

    # perform filtering:
    iterations_reg = 3000
    filtered3D = filter3D(vol3d, iterations_reg, DEVICE_no)
# %%

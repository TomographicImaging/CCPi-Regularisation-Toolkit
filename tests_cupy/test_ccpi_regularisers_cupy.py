import time

import numpy as np
import pytest
cp = pytest.importorskip("cupy") # skipping the tests bellow if the CuPy is not installed
from ccpi.filters.regularisersCuPy import ROF_TV as ROF_TV_cupy

from numpy.testing import assert_allclose, assert_equal

from ccpi.filters.regularisers import ROF_TV

eps = 1e-6

@cp.testing.gpu
def test_median_filter2d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data[60,:,:])
    input = cp.asarray(input_np, order="C")
    filtered_data = ROF_TV_cupy(input,
                    regularisation_parameter=0.06,
                    number_of_iterations = 1000,
                    time_marching_parameter=0.001)
    filtered_data = filtered_data.get()
    
    (filtered_data2, infog) = ROF_TV(input_np, 0.06, 1000, 0.001, 0.0, 'gpu')

    assert_allclose(
        np.median(filtered_data), np.median(filtered_data2), rtol=1e-6
    )
    assert filtered_data.dtype == np.float32

@cp.testing.gpu
def test_median_filter3d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data)
    gpu_index = 0 
    with cp.cuda.Device(gpu_index):
    	input_cp = cp.asarray(input_np, order="C")
    filtered_data = ROF_TV_cupy(input_cp,
                           regularisation_parameter=0.06,
                           number_of_iterations = 1000,
                           time_marching_parameter=0.001,
                           gpu_id = gpu_index)
    filtered_data = filtered_data.get()
    
    (filtered_data2, infog) = ROF_TV(input_np, 0.06, 1000, 0.001, 0.0, 'gpu')

    assert_allclose(
        np.median(filtered_data), np.median(filtered_data2), rtol=1e-6
    )
    assert filtered_data.dtype == np.float32

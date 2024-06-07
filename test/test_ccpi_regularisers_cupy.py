import time
import numpy as np
import cupy as cp
import pytest
from ccpi.filters.regularisersCuPy import ROF_TV, PD_TV

from numpy.testing import assert_allclose, assert_equal

eps = 1e-5
def test_ROF_TV_2d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data[60,:,:])
    input_cp = cp.asarray(input_np, order="C")  
    filtered_data = ROF_TV(input_cp,
                            regularisation_parameter=0.06,
                            iterations = 1000,
                            time_marching_parameter=0.001,
                            gpu_id = 0)
    filtered_data = filtered_data.get()

    # comparison to the CUDA implemented output
    assert_allclose(np.sum(filtered_data), 16589834.0, rtol=eps)
    assert_allclose(np.median(filtered_data), 960.0731, rtol=eps)
    assert filtered_data.dtype == np.float32

def test_ROF_TV_3d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data)
    input_cp = cp.asarray(input_np, order="C")
    filtered_data = ROF_TV(input_cp,
                            regularisation_parameter=0.06,
                            iterations = 1000,
                            time_marching_parameter=0.001,
                            gpu_id = 0)
    filtered_data = filtered_data.get()

    # comparison to the CUDA implemented output
    assert_allclose(np.sum(filtered_data), 2982482200.0, rtol=eps)
    assert_allclose(np.median(filtered_data), 960.1991, rtol=eps)
    assert filtered_data.dtype == np.float32

def test_PD_TV_2d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data[60,:,:])
    input_cp = cp.asarray(input_np, order="C")
    filtered_data = PD_TV(input_cp,
                            regularisation_parameter=0.06,
                            iterations = 1000,
                            tolerance_param = 0.0,
                            methodTV=0,
                            nonneg=0,
                            lipschitz_const=8,
                            gpu_id = 0)    
    filtered_data = filtered_data.get()
    
    # comparison to the CUDA implemented output
    assert_allclose(np.sum(filtered_data), 16589830.0, rtol=eps)
    assert_allclose(np.median(filtered_data), 960.11084, rtol=eps)
    assert filtered_data.dtype == np.float32

def test_PD_TV_3d(host_data):
    cp.get_default_memory_pool().free_all_blocks()
    input_np = np.float32(host_data)
    input_cp = cp.asarray(input_np, order="C")
    filtered_data = PD_TV(input_cp,
                            regularisation_parameter=0.06,
                            iterations = 1000,
                            tolerance_param = 0.0,
                            methodTV=0,
                            nonneg=0,
                            lipschitz_const=8,
                            gpu_id = 0)
    filtered_data = filtered_data.get()
    
    # comparison to the CUDA implemented output
    assert_allclose(np.sum(filtered_data), 2982481000.0, rtol=eps)
    assert_allclose(np.median(filtered_data), 960.1847, rtol=eps)
    assert filtered_data.dtype == np.float32
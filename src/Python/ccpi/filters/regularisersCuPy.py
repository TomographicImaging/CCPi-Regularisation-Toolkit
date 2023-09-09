# This work is part of the Core Imaging Library developed by
# Visual Analytics and Imaging System Group of the Science Technology
# Facilities Council, STFC

# Copyright 2019 Daniil Kazantsev
# Copyright 2019 Srikanth Nagella, Edoardo Pasca

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Regularisation functions are exposed through CuPy API """

try:
    import cupy as cp
except ImportError:
    print("CuPy is required, please install")
import numpy as np

from typing import Optional, Tuple
from ccpi.cuda_kernels import load_cuda_module

__all__ = [
    "ROF_TV",
    "PD_TV",
]


def ROF_TV(data: cp.ndarray,
           regularisation_parameter: Optional[float] = 1e-05,
           number_of_iterations: Optional[int] = 3000,
           time_marching_parameter: Optional[float] = 0.001,
           gpu_id: Optional[int] = 0,
           ) -> cp.ndarray:
    """Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.      
       This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations. 
       Ref: Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms", 1992.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control smoothing. Defaults to 1e-05.
        number_of_iterations (Optional[int], optional): The number of iterations. Defaults to 3000.
        time_marching_parameter (Optional[float], optional): Time marching parameter, needs to be small to ensure convergance. Defaults to 0.001.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

    Returns:
        cp.ndarray: A ROF-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")
    cp.get_default_memory_pool().free_all_blocks()
    
    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    # initialise CuPy arrays here
    out = data.copy()
    d_D1 = cp.empty(data.shape, dtype=cp.float32, order="C")
    d_D2 = cp.empty(data.shape, dtype=cp.float32, order="C")
     
    # loading and compiling CUDA kernels:
    module = load_cuda_module("TV_ROF_GPU_kernels")
    if data.ndim == 3:
        data3d = True
        d_D3 = cp.empty(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)        
        D1_func = module.get_function("D1_func3D")
        D2_func = module.get_function("D2_func3D")
        D3_func = module.get_function("D3_func3D")        
        TV_kernel = module.get_function("TV_kernel3D")        
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        D1_func = module.get_function("D1_func2D")
        D2_func = module.get_function("D2_func2D")
        TV_kernel = module.get_function("TV_kernel2D")    
    
    # perform algorithm iterations
    for iter in range(number_of_iterations):
        # calculate differences        
        if data3d:
            params1 = (out, d_D1, dx, dy, dz)
        else:
            params1 = (out, d_D1, dx, dy)
        D1_func(grid_dims, block_dims, params1)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params2 = (out, d_D2, dx, dy, dz)
        else:
            params2 = (out, d_D2, dx, dy)
        D2_func(grid_dims, block_dims, params2)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params21 = (out, d_D3, dx, dy, dz)
            D3_func(grid_dims, block_dims, params21)
            cp.cuda.runtime.deviceSynchronize()
        # calculating the divergence and the gradient term
        if data3d:
            params3 = (d_D1, d_D2, d_D3, out, data, cp.float32(regularisation_parameter), cp.float32(time_marching_parameter), dx, dy, dz)
        else:
            params3 = (d_D1, d_D2, out, data, cp.float32(regularisation_parameter), cp.float32(time_marching_parameter), dx, dy)
        TV_kernel(grid_dims, block_dims, params3)
        cp.cuda.runtime.deviceSynchronize()
    return out


def PD_TV(data: cp.ndarray,
           regularisation_parameter: Optional[float] = 1e-05,
           number_of_iterations: Optional[int] = 3000,
           time_marching_parameter: Optional[float] = 0.001,
           gpu_id: Optional[int] = 0,
           ) -> cp.ndarray:
    """Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.      
       This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations. 
       Ref: Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms", 1992.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control smoothing. Defaults to 1e-05.
        number_of_iterations (Optional[int], optional): The number of iterations. Defaults to 3000.
        time_marching_parameter (Optional[float], optional): Time marching parameter, needs to be small to ensure convergance. Defaults to 0.001.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

    Returns:
        cp.ndarray: A ROF-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")
    cp.get_default_memory_pool().free_all_blocks()
    
    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    # initialise CuPy arrays here
    out = data.copy()
    d_D1 = cp.empty(data.shape, dtype=cp.float32, order="C")
    d_D2 = cp.empty(data.shape, dtype=cp.float32, order="C")
     
    # loading and compiling CUDA kernels:
    module = load_cuda_module("TV_ROF_GPU_kernels")
    if data.ndim == 3:
        data3d = True
        d_D3 = cp.empty(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)        
        D1_func = module.get_function("D1_func3D")
        D2_func = module.get_function("D2_func3D")
        D3_func = module.get_function("D3_func3D")        
        TV_kernel = module.get_function("TV_kernel3D")        
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        D1_func = module.get_function("D1_func2D")
        D2_func = module.get_function("D2_func2D")
        TV_kernel = module.get_function("TV_kernel2D")    
    
    # perform algorithm iterations
    for iter in range(number_of_iterations):
        # calculate differences        
        if data3d:
            params1 = (out, d_D1, dx, dy, dz)
        else:
            params1 = (out, d_D1, dx, dy)
        D1_func(grid_dims, block_dims, params1)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params2 = (out, d_D2, dx, dy, dz)
        else:
            params2 = (out, d_D2, dx, dy)
        D2_func(grid_dims, block_dims, params2)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params21 = (out, d_D3, dx, dy, dz)
            D3_func(grid_dims, block_dims, params21)
            cp.cuda.runtime.deviceSynchronize()
        # calculating the divergence and the gradient term
        if data3d:
            params3 = (d_D1, d_D2, d_D3, out, data, cp.float32(regularisation_parameter), cp.float32(time_marching_parameter), dx, dy, dz)
        else:
            params3 = (d_D1, d_D2, out, data, cp.float32(regularisation_parameter), cp.float32(time_marching_parameter), dx, dy)
        TV_kernel(grid_dims, block_dims, params3)
        cp.cuda.runtime.deviceSynchronize()
    return out
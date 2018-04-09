# CCPi-Regularization Toolkit (CCPi-RGL)

**Iterative image reconstruction (IIR) methods normally require regularization to stabilize the convergence and make the reconstruction problem more well-posed. 
CCPi-RGL software consist of 2D/3D regularization modules for single-channel and multi-channel reconstruction problems. The modules especially suited for IIR, however,
can also be used as image denoising iterative filters. The core modules are written in C-OMP and CUDA languages and wrappers for Matlab and Python are provided.** 

## Prerequisites: 

 * MATLAB (www.mathworks.com/products/matlab/) OR
 * Python (tested ver. 3.5); Cython
 * C compilers
 * nvcc (CUDA SDK) compilers

## Package modules (regularisers):

### Single-channel
1. Rudin-Osher-Fatemi (ROF) Total Variation (explicit PDE minimisation scheme) [2D/3D GPU/CPU]; (Ref. 1)
2. Fast-Gradient-Projection (FGP) Total Variation [2D/3D GPU/CPU]; (Ref. 2)

### Multi-channel

## Installation:

### Python (conda-build)
```
	export CIL_VERSION=0.9.2
	conda build recipes/regularizers --numpy 1.12 --python 3.5 
	conda install cil_regularizer=0.9.2 --use-local --force
	cd Wrappers/Python
	conda build conda-recipe --numpy 1.12 --python 3.5 
	conda install ccpi-regularizer=0.9.2 --use-local --force
	cd test/
	python test_cpu_vs_gpu_regularizers.py
```
### Matlab
```
	cd /Wrappers/Matlab/mex_compile
	compileCPU_mex.m % to compile CPU modules
	compileGPU_mex.m % to compile GPU modules (see instructions in the file)
```

### References:
1. Rudin, L.I., Osher, S. and Fatemi, E., 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), pp.259-268.
2. Beck, A. and Teboulle, M., 2009. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. IEEE Transactions on Image Processing, 18(11), pp.2419-2434.
3. Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.

### License:
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Acknowledgments:
CCPi-RGL software is a product of the [CCPi](https://www.ccpi.ac.uk/) group and STFC SCD software developers. Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com


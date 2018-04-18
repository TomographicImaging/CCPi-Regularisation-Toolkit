# CCPi-Regularisation Toolkit (CCPi-RGL)

**Iterative image reconstruction (IIR) methods normally require regularisation to stabilise the convergence and make the reconstruction problem more well-posed. 
CCPi-RGL software consist of 2D/3D regularisation modules for single-channel and multi-channel reconstruction problems. The regularisation modules are well-suited for 
[splitting algorithms](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers), of ADMM or FISTA type. Furthermore, 
the toolkit can be used independently to solve image denoising problems. The core modules are written in C-OMP and CUDA languages and wrappers for Matlab and Python are provided.** 

<div align="center">
  <img src="docs/images/probl.png" height="225"><br>  
</div>

## Prerequisites: 

 * [MATLAB](www.mathworks.com/products/matlab/) OR
 * Python (tested ver. 3.5); Cython
 * C compilers
 * nvcc (CUDA SDK) compilers

## Package modules (regularisers):

### Single-channel
1. Rudin-Osher-Fatemi (ROF) Total Variation (explicit PDE minimisation scheme) **2D/3D CPU/GPU** (Ref. *1*)
2. Fast-Gradient-Projection (FGP) Total Variation **2D/3D CPU/GPU** (Ref. *2*)
3. Split-Bregman (SB) Total Variation **2D/3D CPU/GPU** (Ref. *4*)
4. Linear and nonlinear diffusion (explicit PDE minimisation scheme) **2D/3D CPU/GPU** (Ref. *6*)

### Multi-channel
1. Fast-Gradient-Projection (FGP) Directional Total Variation **2D/3D CPU/GPU** (Ref. *3,2*)
2. Total Nuclear Variation (TNV) penalty **2D+channels CPU** (Ref. *5*)

## Installation:

### Python (conda-build)
```
	export CIL_VERSION=0.9.2
	conda build recipes/regularisers --numpy 1.12 --python 3.5 
	conda install cil_regulariser=0.9.2 --use-local --force
	cd Wrappers/Python
	conda build conda-recipe --numpy 1.12 --python 3.5 
	conda install ccpi-regulariser=0.9.2 --use-local --force
	cd demos/
	python demo_cpu_regularisers.py # to run CPU demo
	python demo_gpu_regularisers.py # to run GPU demo
```
### Matlab
```
	cd /Wrappers/Matlab/mex_compile
	compileCPU_mex.m % to compile CPU modules
	compileGPU_mex.m % to compile GPU modules (see instructions in the file)
```

### References:
*1. Rudin, L.I., Osher, S. and Fatemi, E., 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), pp.259-268.*

*2. Beck, A. and Teboulle, M., 2009. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. IEEE Transactions on Image Processing, 18(11), pp.2419-2434.*

*3. Ehrhardt, M.J. and Betcke, M.M., 2016. Multicontrast MRI reconstruction with structure-guided total variation. SIAM Journal on Imaging Sciences, 9(3), pp.1084-1106.*

*4. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.*

*5. Duran, J., Moeller, M., Sbert, C. and Cremers, D., 2016. Collaborative total variation: a general framework for vectorial TV models. SIAM Journal on Imaging Sciences, 9(1), pp.116-151.*

*6. Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.*

### License:
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Acknowledgments:
CCPi-RGL software is a product of the [CCPi](https://www.ccpi.ac.uk/) group and STFC SCD software developers. Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com

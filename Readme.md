# CCPi-Regularisation Toolkit (CCPi-RGL)

**Iterative image reconstruction (IIR) methods normally require regularisation to stabilise the convergence and make the reconstruction problem more well-posed. CCPi-RGL software consists of 2D/3D regularisation modules for single-channel and multi-channel reconstruction problems. The regularisation modules are well-suited to use with [splitting algorithms](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#Alternating_direction_method_of_multipliers), such as ADMM and FISTA. Furthermore, the toolkit can be used independently to solve image denoising and inpaiting tasks. The core modules are written in C-OMP and CUDA languages, wrappers for Matlab and Python are provided.** 

<div align="center">
  <img src="docs/images/probl.png" height="225"><br>  
</div>

## Prerequisites: 

 * [MATLAB](www.mathworks.com/products/matlab/) OR
 * Python (tested ver. 3.5); Cython
 * C compilers
 * nvcc (CUDA SDK) compilers

## Package modules:

### Single-channel (denoising):
1. Rudin-Osher-Fatemi (ROF) Total Variation (explicit PDE minimisation scheme) **2D/3D CPU/GPU** (Ref. *1*)
2. Fast-Gradient-Projection (FGP) Total Variation **2D/3D CPU/GPU** (Ref. *2*)
3. Split-Bregman (SB) Total Variation **2D/3D CPU/GPU** (Ref. *5*)
4. Total Generalised Variation (TGV) model for higher-order regularisation **2D CPU/GPU** (Ref. *6*)
5. Linear and nonlinear diffusion (explicit PDE minimisation scheme) **2D/3D CPU/GPU** (Ref. *8*)
6. Anisotropic Fourth-Order Diffusion (explicit PDE minimisation) **2D/3D CPU/GPU** (Ref. *9*)
7. A joint ROF-LLT (Lysaker-Lundervold-Tai) model for higher-order regularisation **2D/3D CPU/GPU** (Ref. *10,11*)

### Multi-channel (denoising):
1. Fast-Gradient-Projection (FGP) Directional Total Variation **2D/3D CPU/GPU** (Ref. *3,4,2*)
2. Total Nuclear Variation (TNV) penalty **2D+channels CPU** (Ref. *7*)

### Inpainting:
1. Linear and nonlinear diffusion (explicit PDE minimisation scheme) **2D/3D CPU** (Ref. *8*)
2. Iterative nonlocal vertical marching method  **2D CPU**


## Installation:

### Python binaries
Python binaries are distributed via the [ccpi](https://anaconda.org/ccpi/ccpi-regulariser) conda channel. Currently we produce packages for Linux64, Python 2.7, 3.5 and 3.6, NumPy 1.12 and 1.13.

```
conda install ccpi-regulariser -c ccpi -c conda-forge
```

### Python (conda-build)
```
	export CIL_VERSION=0.9.4
	conda build recipes/regularisers --numpy 1.12 --python 3.5 
	conda install cil_regulariser=0.9.4 --use-local --force
	cd Wrappers/Python
	conda build conda-recipe --numpy 1.12 --python 3.5 --no-test
	conda install ccpi-regulariser=0.9.4 --use-local --force
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

### References to implemented methods:
1. [Rudin, L.I., Osher, S. and Fatemi, E., 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), pp.259-268.](https://www.sciencedirect.com/science/article/pii/016727899290242F)

2. [Beck, A. and Teboulle, M., 2009. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. IEEE Transactions on Image Processing, 18(11), pp.2419-2434.](https://doi.org/10.1109/TIP.2009.2028250)

3. [Ehrhardt, M.J. and Betcke, M.M., 2016. Multicontrast MRI reconstruction with structure-guided total variation. SIAM Journal on Imaging Sciences, 9(3), pp.1084-1106.](https://doi.org/10.1137/15M1047325)

4. [Kazantsev, D., Jørgensen, J.S., Andersen, M., Lionheart, W.R., Lee, P.D. and Withers, P.J., 2018. Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography. Inverse Problems, 34(6)](https://doi.org/10.1088/1361-6420/aaba86) **Results can be reproduced using the following** [SOFTWARE](https://github.com/dkazanc/multi-channel-X-ray-CT)

5. [Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.](https://doi.org/10.1137/080725891)

6. [Bredies, K., Kunisch, K. and Pock, T., 2010. Total generalized variation. SIAM Journal on Imaging Sciences, 3(3), pp.492-526.](https://doi.org/10.1137/090769521)

7. [Duran, J., Moeller, M., Sbert, C. and Cremers, D., 2016. Collaborative total variation: a general framework for vectorial TV models. SIAM Journal on Imaging Sciences, 9(1), pp.116-151.](https://doi.org/10.1137/15M102873X)

8. [Black, M.J., Sapiro, G., Marimont, D.H. and Heeger, D., 1998. Robust anisotropic diffusion. IEEE Transactions on image processing, 7(3), pp.421-432.](https://doi.org/10.1109/83.661192)

9. [Hajiaboli, M.R., 2011. An anisotropic fourth-order diffusion filter for image noise removal. International Journal of Computer Vision, 92(2), pp.177-191.](https://doi.org/10.1007/s11263-010-0330-1)

10. [Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.](https://doi.org/10.1109/TIP.2003.819229)

11. [Kazantsev, D., Guo, E., Phillion, A.B., Withers, P.J. and Lee, P.D., 2017. Model-based iterative reconstruction using higher-order regularization of dynamic synchrotron data. Measurement Science and Technology, 28(9), p.094004.](https://doi.org/10.1088/1361-6501/aa7fa8)

### References to Software:
* If software is used, please refer to [11], however, the supporting publication is in progress. 

### Applications:

* [Regularised FISTA-type iterative reconstruction algorithm for X-ray tomographic reconstruction with highly inaccurate measurements (MATLAB code)](https://github.com/dkazanc/FISTA-tomo)
* [Joint image reconstruction method with correlative multi-channel prior for X-ray spectral computed tomography](https://github.com/dkazanc/multi-channel-X-ray-CT)

### License:
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Acknowledgments:
CCPi-RGL software is a product of the [CCPi](https://www.ccpi.ac.uk/) group and STFC SCD software developers. Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com

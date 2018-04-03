# CCPi-Regularisation Toolkit (CCPi-RUT)

**Iterative image reconstruction (IIR) methods normally require regularisation to stabilise convergence and make the reconstruction problem more well-posed. 
CCPi-RUT software consist of 2D/3D regularisation modules which frequently used for IIR. 
The core modules are written in C-OMP and CUDA languages and wrappers for Matlab and Python are provided.** 

## Prerequisites: 

 * MATLAB (www.mathworks.com/products/matlab/)
 * Python (ver. 3.5); Cython
 * C compilers
 * nvcc (CUDA SDK) compilers

## Package modules (regularisers):

1. Rudin-Osher-Fatemi Total Variation (explicit PDE minimisation scheme) [2D/3D GPU/CPU] (1)
2. Fast-Gradient-Projection Total Variation [2D/3D GPU/CPU] (2)

### Installation:

#### Python (conda-build)
```
     export CIL_VERSION=0.9.2
```
#### Matlab 

### References:
1. Rudin, L.I., Osher, S. and Fatemi, E., 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), pp.259-268.
2. Beck, A. and Teboulle, M., 2009. Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. IEEE Transactions on Image Processing, 18(11), pp.2419-2434.
3. Lysaker, M., Lundervold, A. and Tai, X.C., 2003. Noise removal using fourth-order partial differential equation with applications to medical magnetic resonance images in space and time. IEEE Transactions on image processing, 12(12), pp.1579-1590.

### License:
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Acknowledgments:
CCPi-RUT software is a product of the [CCPi](https://www.ccpi.ac.uk/) group and STFC SCD software developers. Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com


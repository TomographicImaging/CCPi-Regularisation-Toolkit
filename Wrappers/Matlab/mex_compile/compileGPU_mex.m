% execute this mex file in Matlab once

%>>>>>>>>>>>>>>Important<<<<<<<<<<<<<<<<<<<
% In order to compile CUDA modules one needs to have nvcc-compiler
% installed (see CUDA SDK)
% check it under MATLAB with !nvcc --version
% In the code bellow we provide a full path to nvcc compiler 
% ! paths to matlab and CUDA sdk can be different, modify accordingly !

% tested on Ubuntu 16.04/MATLAB 2016b

copyfile ../../../Core/regularisers_GPU/ regularisers_GPU/
copyfile ../../../Core/CCPiDefines.h regularisers_GPU/

cd regularisers_GPU/

fprintf('%s \n', 'Compiling GPU regularisers (CUDA)...');
!/usr/local/cuda/bin/nvcc -O0 -c TV_ROF_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu ROF_TV_GPU.cpp TV_ROF_GPU_core.o
movefile ROF_TV_GPU.mex* ../installed/

!/usr/local/cuda/bin/nvcc -O0 -c TV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu FGP_TV_GPU.cpp TV_FGP_GPU_core.o
movefile FGP_TV_GPU.mex* ../installed/

!/usr/local/cuda/bin/nvcc -O0 -c dTV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu FGP_dTV_GPU.cpp dTV_FGP_GPU_core.o
movefile FGP_dTV_GPU.mex* ../installed/

delete TV_ROF_GPU_core* TV_FGP_GPU_core* dTV_FGP_GPU_core* CCPiDefines.h
fprintf('%s \n', 'All successfully compiled!');

cd ../../
cd demos

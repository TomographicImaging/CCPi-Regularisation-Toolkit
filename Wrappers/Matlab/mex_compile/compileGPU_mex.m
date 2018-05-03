% execute this mex file in Matlab once

%>>>>>>>>>>>>>>>>>Important<<<<<<<<<<<<<<<<<<<
% In order to compile CUDA modules one needs to have nvcc-compiler
% installed (see CUDA SDK), check it under MATLAB with !nvcc --version

% In the code bellow we provide a full explicit path to nvcc compiler 
% ! paths to matlab and CUDA sdk can be different, modify accordingly !

% Tested on Ubuntu 16.04/MATLAB 2016b/cuda7.5/gcc4.9
% It hasn't been tested on Windows, please contact me if you'll be able to
% install it on Windows and I include it into the release. 

pathcopyFrom = sprintf(['..' filesep '..' filesep '..' filesep 'Core' filesep 'regularisers_GPU'], 1i);
pathcopyFrom1 = sprintf(['..' filesep '..' filesep '..' filesep 'Core' filesep 'CCPiDefines.h'], 1i);

copyfile(pathcopyFrom, 'regularisers_GPU');
copyfile(pathcopyFrom1, 'regularisers_GPU');

cd regularisers_GPU

Pathmove = sprintf(['..' filesep 'installed' filesep], 1i);

fprintf('%s \n', 'Compiling GPU regularisers (CUDA)...');
!/usr/local/cuda/bin/nvcc -O0 -c TV_ROF_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu ROF_TV_GPU.cpp TV_ROF_GPU_core.o
movefile('ROF_TV_GPU.mex*',Pathmove);

!/usr/local/cuda/bin/nvcc -O0 -c TV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu FGP_TV_GPU.cpp TV_FGP_GPU_core.o
movefile('FGP_TV_GPU.mex*',Pathmove);

!/usr/local/cuda/bin/nvcc -O0 -c TV_SB_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu SB_TV_GPU.cpp TV_SB_GPU_core.o
movefile('SB_TV_GPU.mex*',Pathmove);

!/usr/local/cuda/bin/nvcc -O0 -c dTV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu FGP_dTV_GPU.cpp dTV_FGP_GPU_core.o
movefile('FGP_dTV_GPU.mex*',Pathmove);

!/usr/local/cuda/bin/nvcc -O0 -c NonlDiff_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu NonlDiff_GPU.cpp NonlDiff_GPU_core.o
movefile('NonlDiff_GPU.mex*',Pathmove);

!/usr/local/cuda/bin/nvcc -O0 -c Diffus_4thO_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-7.5/include -L/usr/local/cuda-7.5/lib64 -lcudart -lcufft -lmwgpu Diffusion_4thO_GPU.cpp Diffus_4thO_GPU_core.o
movefile('Diffusion_4thO_GPU.mex*',Pathmove);

delete TV_ROF_GPU_core* TV_FGP_GPU_core* TV_SB_GPU_core* dTV_FGP_GPU_core* NonlDiff_GPU_core* Diffus_4thO_GPU_core* CCPiDefines.h
fprintf('%s \n', 'All successfully compiled!');

pathA2 = sprintf(['..' filesep '..' filesep], 1i);
cd(pathA2);
cd demos
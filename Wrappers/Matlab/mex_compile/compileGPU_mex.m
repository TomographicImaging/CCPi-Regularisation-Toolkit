% execute this mex file in Matlab once

%>>>>>>>>>>>>>>>>>Important<<<<<<<<<<<<<<<<<<<
% In order to compile CUDA modules one needs to have nvcc-compiler
% installed (see CUDA SDK), check it under MATLAB with !nvcc --version

% In the code bellow we provide a full explicit path to nvcc compiler 
% ! paths to matlab and CUDA sdk can be different, modify accordingly !

% Tested on Ubuntu 18.04/MATLAB 2016b/cuda10.0/gcc7.3

% Installation HAS NOT been tested on Windows, please you Cmake build or
% modify the code bellow accordingly
fsep = '/';

pathcopyFrom = sprintf(['..' fsep '..' fsep '..' fsep 'Core' fsep 'regularisers_GPU'], 1i);
pathcopyFrom1 = sprintf(['..' fsep '..' fsep '..' fsep 'Core' fsep 'CCPiDefines.h'], 1i);

copyfile(pathcopyFrom, 'regularisers_GPU');
copyfile(pathcopyFrom1, 'regularisers_GPU');

cd regularisers_GPU

Pathmove = sprintf(['..' fsep 'installed' fsep], 1i);

fprintf('%s \n', '<<<<<<<<<<<Compiling GPU regularisers (CUDA)>>>>>>>>>>>>>');

fprintf('%s \n', 'Compiling ROF-TV...');
!/usr/local/cuda/bin/nvcc -O0 -c TV_ROF_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu ROF_TV_GPU.cpp TV_ROF_GPU_core.o
movefile('ROF_TV_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling FGP-TV...');
!/usr/local/cuda/bin/nvcc -O0 -c TV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu FGP_TV_GPU.cpp TV_FGP_GPU_core.o
movefile('FGP_TV_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling SB-TV...');
!/usr/local/cuda/bin/nvcc -O0 -c TV_SB_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu SB_TV_GPU.cpp TV_SB_GPU_core.o
movefile('SB_TV_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling TGV...');
!/usr/local/cuda/bin/nvcc -O0 -c TGV_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu TGV_GPU.cpp TGV_GPU_core.o
movefile('TGV_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling dFGP-TV...');
!/usr/local/cuda/bin/nvcc -O0 -c dTV_FGP_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu FGP_dTV_GPU.cpp dTV_FGP_GPU_core.o
movefile('FGP_dTV_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling NonLinear Diffusion...');
!/usr/local/cuda/bin/nvcc -O0 -c NonlDiff_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu NonlDiff_GPU.cpp NonlDiff_GPU_core.o
movefile('NonlDiff_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling Anisotropic diffusion of higher order...');
!/usr/local/cuda/bin/nvcc -O0 -c Diffus_4thO_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu Diffusion_4thO_GPU.cpp Diffus_4thO_GPU_core.o
movefile('Diffusion_4thO_GPU.mex*',Pathmove);

fprintf('%s \n', 'Compiling ROF-LLT...');
!/usr/local/cuda/bin/nvcc -O0 -c LLT_ROF_GPU_core.cu -Xcompiler -fPIC -I~/SOFT/MATLAB9/extern/include/
mex -g -I/usr/local/cuda-10.0/include -L/usr/local/cuda-10.0/lib64 -lcudart -lcufft -lmwgpu LLT_ROF_GPU.cpp LLT_ROF_GPU_core.o
movefile('LLT_ROF_GPU.mex*',Pathmove);


delete TV_ROF_GPU_core* TV_FGP_GPU_core* TV_SB_GPU_core* dTV_FGP_GPU_core* NonlDiff_GPU_core* Diffus_4thO_GPU_core* TGV_GPU_core* LLT_ROF_GPU_core* CCPiDefines.h
fprintf('%s \n', 'All successfully compiled!');

pathA2 = sprintf(['..' fsep '..' fsep], 1i);
cd(pathA2);
cd demos
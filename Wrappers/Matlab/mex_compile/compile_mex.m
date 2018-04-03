% compile mex's in Matlab once
copyfile ../../../Core/regularizers_CPU/ regularizers_CPU/
copyfile ../../../Core/regularizers_GPU/ regularizers_GPU/
copyfile ../../../Core/CCPiDefines.h regularizers_CPU/

cd regularizers_CPU/

% compile C regularizers

mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"

delete ROF_TV_core.c ROF_TV_core.h FGP_TV_core.c FGP_TV_core.h utils.c utils.h CCPiDefines.h

% compile CUDA-based regularizers
%cd regularizers_GPU/

cd ../../
cd demos

% compile mex's in Matlab once
copyfile ../../../Core/regularizers_CPU/ regularizers_CPU/
copyfile ../../../Core/regularizers_GPU/ regularizers_GPU/
copyfile ../../../Core/CCPiDefines.h regularizers_CPU/

cd regularizers_CPU/

% compile C regularizers

mex LLT_model.c LLT_model_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex SplitBregman_TV.c SplitBregman_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex TGV_PD.c TGV_PD_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
mex PatchBased_Regul.c PatchBased_Regul_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"

delete LLT_model_core.c LLT_model_core.h FGP_TV_core.c FGP_TV_core.h SplitBregman_TV_core.c SplitBregman_TV_core.h  TGV_PD_core.c  TGV_PD_core.h PatchBased_Regul_core.c PatchBased_Regul_core.h utils.c utils.h CCPiDefines.h

% compile CUDA-based regularizers
%cd regularizers_GPU/

cd ../../
cd demos

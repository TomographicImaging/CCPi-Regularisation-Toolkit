% execute this mex file in Matlab once
copyfile ../../../Core/regularizers_CPU/ regularizers_CPU/
copyfile ../../../Core/CCPiDefines.h regularizers_CPU/

cd regularizers_CPU/

fprintf('%s \n', 'Compiling CPU regularizers...');
mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile ROF_TV.mex* ../installed/

mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile FGP_TV.mex* ../installed/

delete ROF_TV_core* FGP_TV_core* utils.c utils.h CCPiDefines.h

fprintf('%s \n', 'All successfully compiled!');

cd ../../
cd demos

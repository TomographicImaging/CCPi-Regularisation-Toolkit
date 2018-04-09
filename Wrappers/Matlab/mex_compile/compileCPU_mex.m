% execute this mex file in Matlab once
copyfile ../../../Core/regularisers_CPU/ regularisers_CPU/
copyfile ../../../Core/CCPiDefines.h regularisers_CPU/

cd regularisers_CPU/

fprintf('%s \n', 'Compiling CPU regularisers...');
mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile ROF_TV.mex* ../installed/

mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile FGP_TV.mex* ../installed/

delete ROF_TV_core* FGP_TV_core* utils.c utils.h CCPiDefines.h

fprintf('%s \n', 'All successfully compiled!');

cd ../../
cd demos

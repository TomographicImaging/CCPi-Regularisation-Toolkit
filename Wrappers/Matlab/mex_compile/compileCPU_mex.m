% execute this mex file in Matlab once
copyfile ../../../Core/regularisers_CPU/ regularisers_CPU/
copyfile ../../../Core/CCPiDefines.h regularisers_CPU/

cd regularisers_CPU/

fprintf('%s \n', 'Compiling CPU regularisers...');
mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile ROF_TV.mex* ../installed/

mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile FGP_TV.mex* ../installed/

mex SB_TV.c SB_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile SB_TV.mex* ../installed/

mex FGP_dTV.c FGP_dTV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile FGP_dTV.mex* ../installed/

mex TNV.c TNV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile TNV.mex* ../installed/

delete SB_TV_core* ROF_TV_core* FGP_TV_core* FGP_dTV_core* TNV_core* utils* CCPiDefines.h

fprintf('%s \n', 'All successfully compiled!');

cd ../../
cd demos

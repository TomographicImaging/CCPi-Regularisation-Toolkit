% execute this mex file in Matlab once

pathcopyFrom = sprintf(['..' filesep '..' filesep '..' filesep 'Core' filesep 'regularisers_CPU'], 1i);
pathcopyFrom1 = sprintf(['..' filesep '..' filesep '..' filesep 'Core' filesep 'CCPiDefines.h'], 1i);

copyfile(pathcopyFrom, 'regularisers_CPU');
copyfile(pathcopyFrom1, 'regularisers_CPU');

cd regularisers_CPU

Pathmove = sprintf(['..' filesep 'installed' filesep], 1i);

fprintf('%s \n', 'Compiling CPU regularisers...');
mex ROF_TV.c ROF_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('ROF_TV.mex*',Pathmove);

mex FGP_TV.c FGP_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('FGP_TV.mex*',Pathmove);

mex SB_TV.c SB_TV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('SB_TV.mex*',Pathmove);

mex FGP_dTV.c FGP_dTV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('FGP_dTV.mex*',Pathmove);

mex TNV.c TNV_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('TNV.mex*',Pathmove);

mex NonlDiff.c Diffusion_core.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('NonlDiff.mex*',Pathmove);

mex TV_energy.c utils.c CFLAGS="\$CFLAGS -fopenmp -Wall -std=c99" LDFLAGS="\$LDFLAGS -fopenmp"
movefile('TV_energy.mex*',Pathmove);

delete SB_TV_core* ROF_TV_core* FGP_TV_core* FGP_dTV_core* TNV_core* utils* Diffusion_core* CCPiDefines.h
fprintf('%s \n', 'All successfully compiled!');

pathA = sprintf(['..' filesep '..' filesep], 1i);
cd(pathA);
cd demos

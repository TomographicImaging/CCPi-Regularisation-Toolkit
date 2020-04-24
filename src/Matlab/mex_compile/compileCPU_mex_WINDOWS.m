% execute this mex file on Windows in Matlab once

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% I've been able to compile on Windows 7 with MinGW and Matlab 2016b, however, 
% not sure if openmp is enabled after the compilation. 

% Here I present two ways how software can be compiled, if you have some
% other suggestions/remarks please contact me at dkazanc@hotmail.com 
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% cuda related messgaes can be  solved in Linux:
%export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
%export CUDA_HOME=/usr/local/cuda
%export LD_PRELOAD=/home/algol/anaconda3/lib/libstdc++.so.6.0.24
% >>
fsep = '/';

pathcopyFrom = sprintf(['..' fsep '..' fsep 'Core' fsep 'regularisers_CPU'], 1i);
pathcopyFrom1 = sprintf(['..' fsep '..' fsep 'Core' fsep 'CCPiDefines.h'], 1i);
pathcopyFrom2 = sprintf(['..' fsep '..' fsep 'Core' fsep 'inpainters_CPU'], 1i);

copyfile(pathcopyFrom, 'regularisers_CPU');
copyfile(pathcopyFrom1, 'regularisers_CPU');
copyfile(pathcopyFrom2, 'regularisers_CPU');

cd regularisers_CPU

Pathmove = sprintf(['..' fsep 'installed' fsep], 1i);

fprintf('%s \n', '<<<<<<<<<<<Compiling CPU regularisers>>>>>>>>>>>>>');

fprintf('%s \n', 'Compiling ROF-TV...');
mex ROF_TV.c ROF_TV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('ROF_TV.mex*',Pathmove);

fprintf('%s \n', 'Compiling FGP-TV...');
mex FGP_TV.c FGP_TV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('FGP_TV.mex*',Pathmove);

fprintf('%s \n', 'Compiling SB-TV...');
mex SB_TV.c SB_TV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('SB_TV.mex*',Pathmove);

fprintf('%s \n', 'Compiling dFGP-TV...');
mex FGP_dTV.c FGP_dTV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('FGP_dTV.mex*',Pathmove);

fprintf('%s \n', 'Compiling TNV...');
mex TNV.c TNV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('TNV.mex*',Pathmove);

fprintf('%s \n', 'Compiling NonLinear Diffusion...');
mex NonlDiff.c Diffusion_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('NonlDiff.mex*',Pathmove);

fprintf('%s \n', 'Compiling Anisotropic diffusion of higher order...');
mex Diffusion_4thO.c Diffus4th_order_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('Diffusion_4thO.mex*',Pathmove);

fprintf('%s \n', 'Compiling TGV...');
mex TGV.c TGV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('TGV.mex*',Pathmove);

fprintf('%s \n', 'Compiling ROF-LLT...');
mex LLT_ROF.c LLT_ROF_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('LLT_ROF.mex*',Pathmove);

fprintf('%s \n', 'Compiling NonLocal-TV...');
mex PatchSelect.c PatchSelect_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
mex Nonlocal_TV.c Nonlocal_TV_core.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('Nonlocal_TV.mex*',Pathmove);
movefile('PatchSelect.mex*',Pathmove);

fprintf('%s \n', 'Compiling additional tools...');
mex TV_energy.c utils.c COMPFLAGS="\$COMPFLAGS -fopenmp -Wall -std=c99"
movefile('TV_energy.mex*',Pathmove);

%%
%%% The second approach to compile using TDM-GCC which follows this
%%% discussion:
%%% https://uk.mathworks.com/matlabcentral/answers/279171-using-mingw-compiler-and-open-mp#comment_359122
%%% 1. Install TDM-GCC independently from http://tdm-gcc.tdragon.net/ (I installed 5.1.0)
%%% Install openmp version: http://sourceforge.net/projects/tdm-gcc/files/TDM-GCC%205%20series/5.1.0-tdm64-1/gcc-5.1.0-tdm64-1-openmp.zip/download
%%% 2. Link til libgomp.a in that installation when compilling your mex file.

%%% assuming you unzipped TDM GCC (OpenMp) in folder TDMGCC on C drive, uncomment
%%% bellow
% fprintf('%s \n', 'Compiling CPU regularisers...');
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" ROF_TV.c ROF_TV_core.c utils.c
% movefile('ROF_TV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" FGP_TV.c FGP_TV_core.c utils.c
% movefile('FGP_TV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" SB_TV.c SB_TV_core.c utils.c
% movefile('SB_TV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" FGP_dTV.c FGP_dTV_core.c utils.c
% movefile('FGP_dTV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" TNV.c TNV_core.c utils.c
% movefile('TNV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" NonlDiff.c Diffusion_core.c utils.c
% movefile('NonlDiff.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" Diffusion_4thO.c Diffus4th_order_core.c utils.c
% movefile('Diffusion_4thO.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" TGV.c TGV_core.c utils.c
% movefile('TGV.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" LLT_ROF.c LLT_ROF_core.c utils.c
% movefile('LLT_ROF.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" PatchSelect.c PatchSelect_core.c utils.c
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" Nonlocal_TV.c Nonlocal_TV_core.c utils.c
% movefile('Nonlocal_TV.mex*',Pathmove);
% movefile('PatchSelect.mex*',Pathmove);
% mex C:\TDMGCC\lib\gcc\x86_64-w64-mingw32\5.1.0\libgomp.a CXXFLAGS="$CXXFLAGS -std=c++11 -fopenmp" TV_energy.c utils.c
% movefile('TV_energy.mex*',Pathmove);


delete SB_TV_core* ROF_TV_core* FGP_TV_core* FGP_dTV_core* TNV_core* utils* Diffusion_core* Diffus4th_order_core* TGV_core* CCPiDefines.h
delete PatchSelect_core* Nonlocal_TV_core*
fprintf('%s \n', 'Regularisers successfully compiled!');


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pathA2 = sprintf(['..' fsep '..' fsep '..' fsep '..' fsep 'demos'], 1i);
cd(pathA2);

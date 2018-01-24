% Demonstration of tomographic 3D reconstruction from X-ray synchrotron
% dataset (dendrites) using various data fidelities
% ! It is advisable not to run the whole script, it will take lots of time to reconstruct the whole 3D data using many algorithms !
clear
close all
%%
% % adding paths
addpath('../data/');
addpath('../main_func/'); addpath('../main_func/regularizers_CPU/'); addpath('../main_func/regularizers_GPU/NL_Regul/'); addpath('../main_func/regularizers_GPU/Diffus_HO/');
addpath('../supp/');

load('DendrRawData.mat') % load raw data of 3D dendritic set
angles_rad = angles*(pi/180); % conversion to radians
det_size = size(data_raw3D,1); % detectors dim
angSize = size(data_raw3D, 2); % angles dim
slices_tot = size(data_raw3D, 3); % no of slices
recon_size = 950; % reconstruction size

Sino3D = zeros(det_size, angSize, slices_tot, 'single'); % log-corrected sino
% normalizing the data
for  jj = 1:slices_tot
    sino = data_raw3D(:,:,jj);
    for ii = 1:angSize
        Sino3D(:,ii,jj) = log((flats_ar(:,jj)-darks_ar(:,jj))./(single(sino(:,ii)) - darks_ar(:,jj)));
    end
end

Sino3D = Sino3D.*1000;
Weights3D = single(data_raw3D); % weights for PW model
clear data_raw3D
%%
% set projection/reconstruction geometry here
proj_geom = astra_create_proj_geom('parallel', 1, det_size, angles_rad);
vol_geom = astra_create_vol_geom(recon_size,recon_size);
%%
fprintf('%s\n', 'Reconstruction using FBP...');
FBP = iradon(Sino3D(:,:,10), angles,recon_size);
figure; imshow(FBP , [0, 3]); title ('FBP reconstruction');

%--------FISTA_REC modular reconstruction alogrithms---------
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-PWLS without regularization...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 18;
params.weights = Weights3D;
params.subsets = 8; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 1;

tic; [X_fista, outputFISTA] = FISTA_REC(params); toc;
figure; imshow(X_fista(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-PWLS reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-PWLS-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 18;
params.Regul_Lambda_FGPTV = 5.0000e+6; % TV regularization parameter for FGP-TV
params.weights = Weights3D;
params.subsets = 8; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_TV, outputTV] = FISTA_REC(params); toc;
figure; imshow(X_fista_TV(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-PWLS-TV reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-GH-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D(:,:,10);
params.iterFISTA  = 18;
params.Regul_Lambda_FGPTV = 5.0000e+6; % TV regularization parameter for FGP-TV
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D(:,:,10);
params.subsets = 8; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 1;

tic; [X_fista_GH_TV, outputGHTV] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TV(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-GH-TV reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-GH-TV-LLT...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 12;
params.Regul_Lambda_FGPTV = 5.0000e+6; % TV regularization parameter for FGP-TV
params.Regul_LambdaLLT = 100;  % regularization parameter for LLT problem
params.Regul_tauLLT = 0.0005; % time-step parameter for the explicit scheme
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D;
params.subsets = 16; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 2;

tic; [X_fista_GH_TVLLT, outputGH_TVLLT] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TVLLT(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-GH-TV-LLT reconstruction');

%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-GH-HigherOrderDiffusion...');
% !GPU version!
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D(:,:,1:5);
params.iterFISTA  = 25;
params.Regul_LambdaDiffHO = 2; % DiffHO regularization parameter 
params.Regul_DiffHO_EdgePar = 0.05; % threshold parameter
params.Regul_Iterations = 150;
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D(:,:,1:5);
params.subsets = 16; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 1;
 
tic; [X_fista_GH_HO, outputHO] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_HO(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-HigherOrderDiffusion reconstruction');

%%
fprintf('%s\n', 'Reconstruction using FISTA-PB...');
% !GPU version!
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D(:,:,1);
params.iterFISTA  = 25;
params.Regul_LambdaPatchBased_GPU = 3; % PB regularization parameter 
params.Regul_PB_h = 0.04; % threhsold parameter
params.Regul_PB_SearchW = 3;
params.Regul_PB_SimilW  = 1;
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D(:,:,1);
params.show = 1;
params.maxvalplot = 2.5; params.slice = 1;
 
tic; [X_fista_GH_PB, outputPB] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_PB(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-PB reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-OS-GH-TGV...');
% still testing...
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 12;
params.Regul_LambdaTGV = 0.5; % TGV regularization parameter 
params.Regul_Iterations = 5;
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D;
params.subsets = 16; % the number of ordered subsets 
params.show = 1;
params.maxvalplot = 2.5; params.slice = 1;

tic; [X_fista_GH_TGV, outputTGV] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TGV(:,:,params.slice) , [0, 2.5]); title ('FISTA-OS-GH-TGV reconstruction');


%%
% fprintf('%s\n', 'Reconstruction using FISTA-Student-TV...');
% clear params
% params.proj_geom = proj_geom; % pass geometry to the function
% params.vol_geom = vol_geom;
% params.sino = Sino3D(:,:,10);
% params.iterFISTA  = 50;
% params.L_const = 0.01; % Lipshitz constant
% params.Regul_LambdaTV = 0.008; % TV regularization parameter for FISTA-TV
% params.fidelity = 'student'; % choosing Student t penalty
% params.weights = Weights3D(:,:,10);
% params.show = 0;
% params.initialize = 1;
% params.maxvalplot = 2.5; params.slice = 1;
% 
% tic; [X_fistaStudentTV] = FISTA_REC(params); toc;
% figure; imshow(X_fistaStudentTV(:,:,1), [0, 2.5]); title ('FISTA-Student-TV reconstruction');
%%

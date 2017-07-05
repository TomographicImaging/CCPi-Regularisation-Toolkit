% Demonstration of tomographic 3D reconstruction from X-ray synchrotron
% dataset (dendrites) using various data fidelities
% warning: can take up to 15-20 minutes to run for the whole 3D data
clear all
close all
%%
% % adding paths
addpath('../data/');
addpath('../main_func/');
addpath('../supp/');

load('DendrRawData.mat') % load raw data of 3D dendritic set
angles_rad = angles*(pi/180); % conversion to radians
size_det = size(data_raw3D,1); % detectors dim
angSize = size(data_raw3D, 2); % angles dim
slices_tot = size(data_raw3D, 3); % no of slices
recon_size = 950; % reconstruction size

Sino3D = zeros(size_det, angSize, slices_tot, 'single'); % log-corrected sino
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
Z_slices = 20;
det_row_count = Z_slices;
proj_geom = astra_create_proj_geom('parallel3d', 1, 1, det_row_count, size_det, angles_rad);
vol_geom = astra_create_vol_geom(recon_size,recon_size,Z_slices);
%%
fprintf('%s\n', 'Reconstruction using FBP...');
FBP = iradon(Sino3D(:,:,10), angles,recon_size);
figure; imshow(FBP , [0, 3]); title ('FBP reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-PWLS without regularization...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.L_const = 7.6789e+08; % found quickly for one slice first
params.iterFISTA  = 30;
params.weights = Weights3D;
params.show = 1;
params.maxvalplot = 2.5; params.slice = 4;

tic; [X_fista, output] = FISTA_REC(params); toc;
figure; imshow(X_fista(:,:,1) , [0, 2.5]); title ('FISTA-PWLS reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-PWLS-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 40;
params.L_const = 7.6789e+08; 
params.Regul_Lambda_FGPTV = 0.005; % TV regularization parameter for FGP-TV
params.weights = Weights3D;
params.show = 1;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_TV(:,:,1) , [0, 2.5]); title ('FISTA-PWLS-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 40;
params.L_const = 7.6789e+08; 
params.Regul_Lambda_FGPTV = 0.005; % TV regularization parameter for FGP-TV
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D;
params.show = 1;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_GH_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TV(:,:,1) , [0, 2.5]); title ('FISTA-GH-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV-LLT...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = Sino3D;
params.iterFISTA  = 40;
params.Regul_Lambda_FGPTV = 0.005; % TV regularization parameter for FGP-TV
params.Regul_LambdaHO = 200;  % regularization parameter for LLT problem
params.Regul_tauLLT = 0.0005; % time-step parameter for the explicit scheme
params.Ring_LambdaR_L1 = 0.002; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.weights = Weights3D;
params.show = 1;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_GH_TVLLT] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TVLLT(:,:,1) , [0, 2.5]); title ('FISTA-GH-TV-LLT reconstruction');
%%
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

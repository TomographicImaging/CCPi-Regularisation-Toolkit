% Demonstration of tomographic reconstruction from neutron tomography
% dataset (basalt sample) using Student t data fidelity 
clear all
close all 

% adding paths
addpath('../data/'); 
addpath('../main_func/'); 
addpath('../supp/'); 

load('sino_basalt.mat') % load real neutron data

size_det = size(sino_basalt, 1); % detector size
angSize = size(sino_basalt,2); % angles dim
recon_size = 650; % reconstruction size

FBP = iradon(sino_basalt, rad2deg(angles),recon_size);
figure; imshow(FBP , [0, 0.45]); title ('FBP reconstruction');
%%
% set projection/reconstruction geometry here
Z_slices = 1;
det_row_count = Z_slices;
proj_geom = astra_create_proj_geom('parallel3d', 1, 1, det_row_count, size_det, angles);
vol_geom = astra_create_vol_geom(recon_size,recon_size,Z_slices);
%%
fprintf('%s\n', 'Reconstruction using FISTA-LS without regularization...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_basalt;
params.iterFISTA  = 50;
params.show = 0;
params.maxvalplot = 0.6; params.slice = 1;

tic; [X_fista] = FISTA_REC(params); toc;
figure; imshow(X_fista , [0, 0.45]); title ('FISTA-LS reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-LS-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_basalt;
params.iterFISTA  = 60;
params.Regul_LambdaTV = 0.0003; % TV regularization parameter
params.show = 0;
params.maxvalplot = 0.6; params.slice = 1;

tic; [X_fista_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_TV , [0, 0.45]); title ('FISTA-LS-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_basalt;
params.iterFISTA  = 60;
params.Regul_LambdaTV = 0.0003; % TV regularization parameter
params.Ring_LambdaR_L1 = 0.001; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 20; % acceleration for ring variable
params.show = 0;
params.maxvalplot = 0.6; params.slice = 1;

tic; [X_fista_GH_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TV , [0, 0.45]); title ('FISTA-GH-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-Student-TV...');
clear params
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_basalt;
params.iterFISTA  = 50;
params.L_const = 3500; % Lipshitz constant
params.Regul_LambdaTV = 0.0003; % TV regularization parameter
params.fidelity = 'student'; % choosing Student t penalty
params.show = 1;
params.initialize = 1; % warm start with SIRT
params.maxvalplot = 0.6; params.slice = 1;

tic; [X_fistaStudentTV] = FISTA_REC(params); toc;
figure; imshow(X_fistaStudentTV , [0, 0.45]); title ('FISTA-Student-TV reconstruction');
%%

fprintf('%s\n', 'Segmentation using OTSU method ...');
level = graythresh(X_fista);
Segm_FISTA = im2bw(X_fista,level);
figure; imshow(Segm_FISTA, []); title ('Segmented FISTA-LS reconstruction');

level = graythresh(X_fista_TV);
Segm_FISTA_TV = im2bw(X_fista_TV,level);
figure; imshow(Segm_FISTA_TV, []); title ('Segmented FISTA-LS-TV reconstruction');

level = graythresh(X_fista_GH_TV);
BW_FISTA_GH_TV = im2bw(X_fista_GH_TV,level);
figure; imshow(BW_FISTA_GH_TV, []); title ('Segmented FISTA-GH-TV reconstruction');

level = graythresh(X_fistaStudentTV);
BW_FISTA_Student_TV = im2bw(X_fistaStudentTV,level);
figure; imshow(BW_FISTA_Student_TV, []); title ('Segmented FISTA-Student-LS reconstruction');
% Demonstration of tomographic 3D reconstruction from X-ray synchrotron 
% dataset (dendrites) using various data fidelities 
% clear all
% close all 
% 
% % adding paths
 addpath('data/'); 
 addpath('main_func/'); 
 addpath('supp/'); 

load('sino3D_dendrites.mat') % load 3D normalized sinogram
angles_rad = angles*(pi/180); % conversion to radians

angSize = size(Sino3D,1); % angles dim
size_det = size(Sino3D, 2); % detector size
recon_size = 850; % reconstruction size

FBP = iradon(Sino3D(:,:,10)', angles,recon_size);
figure; imshow(FBP , [0, 3]); title ('FBP reconstruction');

%%
fprintf('%s\n', 'Reconstruction using FISTA-LS without regularization...');
clear params
params.sino = Sino3D;
params.N  = recon_size;
params.angles = angles_rad;
params.iterFISTA  = 80;
params.precondition  = 1; % switch on preconditioning
params.show = 0;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista] = FISTA_REC(params); toc;
figure; imshow(X_fista(:,:,10) , [0, 2.5]); title ('FISTA-LS reconstruction');
%%
fprintf('%s\n', 'Reconstruction using FISTA-LS-TV...');
clear params
params.sino = Sino3D;
params.N  = recon_size;
params.angles = angles_rad;
params.iterFISTA  = 100;
params.lambdaTV = 0.001; % TV regularization parameter for FISTA-TV
params.tol = 1.0e-04;
params.iterTV = 20;
params.precondition  = 1; % switch on preconditioning
params.show = 0;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_TV(:,:,10) , [0, 2.5]); title ('FISTA-LS-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV...');
clear params
params.sino = Sino3D;
params.N  = recon_size;
params.angles = angles_rad;
params.iterFISTA  = 100;
params.lambdaTV = 0.001; % TV regularization parameter for FISTA-TV
params.tol = 1.0e-04;
params.iterTV = 20;
params.lambdaR_L1 = 0.001; % Soft-Thresh L1 ring variable parameter
params.alpha_ring = 20; % to boost ring removal procedure
params.precondition  = 1; % switch on preconditioning
params.show = 0;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_GH_TV] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TV(:,:,10) , [0, 2.5]); title ('FISTA-GH-TV reconstruction');
%%
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV-LLT...');
clear params
params.sino = Sino3D;
params.N  = recon_size;
params.angles = angles_rad;
params.iterFISTA  = 100;
params.lambdaTV = 0.001; % TV regularization parameter for FISTA-TV
params.tol = 1.0e-04;
params.iterTV = 20;
params.lambdaHO = 35;  % regularization parameter for LLT problem
params.tauHO = 0.00011; % time-step parameter for explicit scheme
params.iterHO = 70; % the max number of TV iterations   
params.lambdaR_L1 = 0.001; % Soft-Thresh L1 ring variable parameter
params.alpha_ring = 20; % to boost ring removal procedure
params.precondition  = 1; % switch on preconditioning
params.show = 0;
params.maxvalplot = 2.5; params.slice = 10;

tic; [X_fista_GH_TVLLT] = FISTA_REC(params); toc;
figure; imshow(X_fista_GH_TVLLT(:,:,10) , [0, 2.5]); title ('FISTA-GH-TV-LLT reconstruction');
%%
%%
% fprintf('%s\n', 'Reconstruction using FISTA-Student-TV...');
% %%%%<<<< Not stable with this dataset! Requires more work >>>> %%%%%
% clear params
% params.sino = Sino3D(:,:,15);
% params.N  = 950;
% params.angles = angles_rad;
% params.iterFISTA  = 150;
% params.L_const = 30; % Lipshitz constant
% params.lambdaTV = 0.009; % TV regularization parameter for FISTA-TV
% params.tol = 1.0e-04;
% params.iterTV = 20;
% params.fidelity = 'student'; % choosing Student t penalty
% % params.precondition  = 1; % switch on preconditioning
% params.show = 1;
% params.maxvalplot = 2.5; params.slice = 1;
% 
% tic; [X_fistaStudentTV] = FISTA_REC(params); toc;
% figure; imshow(X_fistaStudentTV , [0, 2.5]); title ('FISTA-Student-TV reconstruction');
%%
slice = 10; % if 3D reconstruction

fprintf('%s\n', 'Segmentation using OTSU method ...');
level = graythresh(X_fista(:,:,slice));
Segm_FISTA = im2bw(X_fista(:,:,slice),level);
figure; imshow(Segm_FISTA, []); title ('Segmented FISTA-LS reconstruction');

level = graythresh(X_fista_TV(:,:,slice));
Segm_FISTA_TV = im2bw(X_fista_TV(:,:,slice),level);
figure; imshow(Segm_FISTA_TV, []); title ('Segmented FISTA-LS-TV reconstruction');

level = graythresh(X_fista_GH_TV(:,:,slice));
BW_FISTA_GH_TV = im2bw(X_fista_GH_TV(:,:,slice),level);
figure; imshow(BW_FISTA_GH_TV, []); title ('Segmented FISTA-GH-TV reconstruction');

level = graythresh(X_fista_GH_TVLLT(:,:,slice));
BW_FISTA_GH_TVLLT = im2bw(X_fista_GH_TVLLT(:,:,slice),level);
figure; imshow(BW_FISTA_GH_TVLLT, []); title ('Segmented FISTA-GH-TV-LLT reconstruction');
%%
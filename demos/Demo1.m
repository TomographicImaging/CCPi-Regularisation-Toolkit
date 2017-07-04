% Demonstration of tomographic reconstruction from noisy and corrupted by 
% artifacts undersampled projection data using Students't penalty 
% Optimisation problem is solved using FISTA algorithm (see Beck & Teboulle)

% see Readme file for instructions
%%
% compile MEX-files ones
% cd ..
% cd main_func
% compile_mex
% cd .. 
% cd demos
%%

close all;clc;clear all;
% adding paths
addpath('../data/'); 
addpath('../main_func/'); 
addpath('../supp/'); 

load phantom_bone512.mat % load the phantom
load my_red_yellowMAP.mat % load the colormap
% load sino1.mat; % load noisy sinogram

N = 512; %  the size of the tomographic image NxN
theta = 1:1:180; % acquisition angles (in parallel beam from 0 to Pi)
theta_rad = theta*(pi/180); % conversion to radians
P = 2*ceil(N/sqrt(2))+1; % the size of the detector array
ROI = find(phantom > 0);

% using ASTRA to set the projection geometry
% potentially parallel geometry can be replaced with a divergent one
Z_slices = 1;
det_row_count = Z_slices;
proj_geom = astra_create_proj_geom('parallel3d', 1, 1, det_row_count, P, theta_rad);
vol_geom = astra_create_vol_geom(N,N,Z_slices);

zing_rings_add;    % generating data, adding zingers and stripes
%% 
fprintf('%s\n', 'Direct reconstruction using FBP...');
FBP_1 = iradon(sino_zing_rings', theta, N); 

fprintf('%s %.4f\n', 'RMSE for FBP reconstruction:', RMSE(FBP_1(:), phantom(:)));

figure(1); 
subplot_tight(1,2,1, [0.05 0.05]); imshow(FBP_1,[0 0.6]);  title('FBP reconstruction of noisy and corrupted by artifacts sinogram'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); imshow((phantom - FBP_1).^2,[0 0.1]);  title('residual: (ideal phantom - FBP)^2'); colorbar;
colormap(cmapnew); 

%%
fprintf('%s\n', 'Reconstruction using FISTA-PWLS without regularization...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_zing_rings; % sinogram
params.iterFISTA = 45; %max number of outer iterations
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
params.weights = Dweights; % statistical weighting 
tic; [X_FISTA, output] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction is:', min(error_FISTA(:)));
error_FISTA = output.Resid_error; obj_FISTA = output.objective;

figure(2); clf
%set(gcf, 'Position', get(0,'Screensize'));
subplot(1,2,1, [0.05 0.05]); imshow(X_FISTA,[0 0.6]); title('FISTA-PWLS reconstruction'); colorbar;
subplot(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 
figure(3); clf
subplot(1,2,1, [0.05 0.05]); plot(error_FISTA);  title('RMSE plot'); colorbar;
subplot(1,2,2, [0.05 0.05]); plot(obj_FISTA);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-PWLS-TV...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_zing_rings;
params.iterFISTA = 45; % max number of outer iterations
params.Regul_LambdaTV = 0.0015; % regularization parameter for TV problem
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
tic; [X_FISTA_TV, output] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS-TV reconstruction is:', min(error_FISTA_TV(:)));
error_FISTA_TV = output.Resid_error; obj_FISTA_TV = output.objective;

figure(4); clf
subplot(1,2,1, [0.05 0.05]); imshow(X_FISTA_TV,[0 0.6]); title('FISTA-PWLS-TV reconstruction'); colorbar;
subplot(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 
figure(5); clf
subplot(1,2,1, [0.05 0.05]); plot(error_FISTA_TV);  title('RMSE plot'); colorbar;
subplot(1,2,2, [0.05 0.05]); plot(obj_FISTA_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_zing_rings;
params.iterFISTA = 50; % max number of outer iterations
params.Regul_LambdaTV = 0.0015;  % regularization parameter for TV problem
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.Ring_LambdaR_L1 = 0.002; % parameter to sparsify the "rings vector"
params.Ring_Alpha = 20; % to accelerate ring-removal procedure
params.show = 0; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
tic; [X_FISTA_GH_TV, output] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-GH-TV reconstruction is:', min(error_FISTA_GH_TV(:)));
error_FISTA_GH_TV = output.Resid_error; obj_FISTA_GH_TV = output.objective;

figure(6); clf
subplot(1,2,1, [0.05 0.05]); imshow(X_FISTA_GH_TV,[0 0.6]); title('FISTA-GH-TV reconstruction'); colorbar;
subplot(1,2,2, [0.05 0.05]);imshow((phantom - X_FISTA_GH_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 

figure(7); clf
subplot(1,2,1, [0.05 0.05]);  plot(error_FISTA_GH_TV);  title('RMSE plot'); colorbar;
subplot(1,2,2, [0.05 0.05]);  plot(obj_FISTA_GH_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-Student-TV...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = sino_zing_rings;
params.iterFISTA = 55; % max number of outer iterations
params.L_const = 0.1; % Lipshitz constant (can be chosen manually to accelerate convergence)
params.Regul_LambdaTV = 0.00152;  % regularization parameter for TV problem
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.fidelity = 'student'; % selecting students t fidelity
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
params.initilize = 1; % warm start with SIRT
tic; [X_FISTA_student_TV, output] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-Student-TV reconstruction is:', min(error_FISTA_student_TV(:)));
error_FISTA_student_TV = output.Resid_error; obj_FISTA_student_TV = output.objective;

figure(8); 
set(gcf, 'Position', get(0,'Screensize'));
subplot(1,2,1, [0.05 0.05]); imshow(X_FISTA_student_TV,[0 0.6]); title('FISTA-Student-TV reconstruction'); colorbar;
subplot(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA_student_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 

figure(9); 
subplot(1,2,1, [0.05 0.05]); plot(error_FISTA_student_TV);  title('RMSE plot'); colorbar;
subplot(1,2,2, [0.05 0.05]); plot(obj_FISTA_student_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
% print all RMSE's
fprintf('%s\n', '--------------------------------------------');
fprintf('%s %.4f\n', 'RMSE for FBP reconstruction:', RMSE(FBP_1(:), phantom(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction:', min(error_FISTA(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS-TV reconstruction:', min(error_FISTA_TV(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-GH-TV reconstruction:', min(error_FISTA_GH_TV(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-Student-TV reconstruction:', min(error_FISTA_student_TV(:)));
%
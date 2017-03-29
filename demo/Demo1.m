% Demonstration of tomographic reconstruction from noisy and corrupted by 
% artifacts undersampled projection data using Students't penalty 
% Optimisation problem is solved using FISTA algorithm (see Beck & Teboulle)

% see ReadMe file for instructions
clear all
close all 

% adding paths
addpath('data/'); 
addpath('main_func/'); 
addpath('supp/'); 

load phantom_bone512.mat % load the phantom
load  my_red_yellowMAP.mat % load the colormap
% load sino1.mat; % load noisy sinogram

N = 512; %  the size of the tomographic image NxN
theta = 1:1:180; % acquisition angles (in parallel beam from 0 to Pi)
theta_rad = theta*(pi/180); % conversion to radians
P = 2*ceil(N/sqrt(2))+1; % the size of the detector array
ROI = find(phantom > 0);

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
fprintf('%s\n', 'Reconstruction using FISTA-LS without regularization...');
clear params
% define parameters
params.sino = sino_zing_rings;
params.N = N; % image size 
params.angles = theta_rad; % angles in radians 
params.iterFISTA = 180; %max number of outer iterations
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.show = 0; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
params.weights = Dweights; % statistical weighting 
tic; [X_FISTA, error_FISTA, obj_FISTA, sinoFISTA] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-LS reconstruction is:', min(error_FISTA(:)));

figure(2); clf
%set(gcf, 'Position', get(0,'Screensize'));
subplot_tight(1,2,1, [0.05 0.05]); imshow(X_FISTA,[0 0.6]); title('FISTA-LS reconstruction'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 
figure(3); clf
subplot_tight(1,2,1, [0.05 0.05]); plot(error_FISTA);  title('RMSE plot'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); plot(obj_FISTA);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-LS-TV...');
clear params
% define parameters
params.sino = sino_zing_rings;
params.N = N; % image size 
params.angles = theta_rad; % angles in radians 
params.iterFISTA = 200; % max number of outer iterations
params.lambdaTV = 5.39e-05; % regularization parameter for TV problem
params.tol = 1.0e-04; % tolerance to terminate TV iterations
params.iterTV = 20; % the max number of TV iterations
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.show = 0; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
tic; [X_FISTA_TV, error_FISTA_TV, obj_FISTA_TV, sinoFISTA_TV] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-LS-TV reconstruction is:', min(error_FISTA_TV(:)));

figure(4); clf
subplot_tight(1,2,1, [0.05 0.05]); imshow(X_FISTA_TV,[0 0.6]); title('FISTA-LS-TV reconstruction'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 
figure(5); clf
subplot_tight(1,2,1, [0.05 0.05]); plot(error_FISTA_TV);  title('RMSE plot'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); plot(obj_FISTA_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-GH-TV...');
clear params
% define parameters
params.sino = sino_zing_rings;
params.N = N; % image size 
params.angles = theta_rad; % angles in radians 
params.iterFISTA = 60; % max number of outer iterations
params.lambdaTV = 0.002526;  % regularization parameter for TV problem
params.tol = 1.0e-04; % tolerance to terminate TV iterations
params.iterTV = 20; % the max number of TV iterations
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.lambdaR_L1 = 0.002; % parameter to sparsify the "rings vector"
params.show = 0; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
tic; [X_FISTA_GH_TV, error_FISTA_GH_TV, obj_FISTA_GH_TV, sinoFISTA_GH_TV] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-GH-TV reconstruction is:', min(error_FISTA_GH_TV(:)));

figure(6); clf
subplot_tight(1,2,1, [0.05 0.05]); imshow(X_FISTA_GH_TV,[0 0.6]); title('FISTA-GH-TV reconstruction'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]);imshow((phantom - X_FISTA_GH_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 

figure(7); clf
subplot_tight(1,2,1, [0.05 0.05]);  plot(error_FISTA_GH_TV);  title('RMSE plot'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]);  plot(obj_FISTA_GH_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
fprintf('%s\n', 'Reconstruction using FISTA-Student-TV...');
clear params
% define parameters
params.sino = sino_zing_rings;
params.N = N; % image size 
params.angles = theta_rad; % angles in radians 
params.iterFISTA = 67; % max number of outer iterations
%params.L_const = 80000; % Lipshitz constant (can be chosen manually to accelerate convergence)
params.lambdaTV = 0.00152;  % regularization parameter for TV problem
params.tol = 1.0e-04; % tolerance to terminate TV iterations
params.iterTV = 20; % the max number of TV iterations
params.X_ideal = phantom; % ideal phantom
params.ROI = ROI; % phantom region-of-interest
params.weights = Dweights; % statistical weighting 
params.fidelity = 'student'; % selecting students t fidelity
params.show = 0; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 0.6; 
tic; [X_FISTA_student_TV, error_FISTA_student_TV, obj_FISTA_student_TV, sinoFISTA_student_TV] = FISTA_REC(params); toc; 

fprintf('%s %.4f\n', 'Min RMSE for FISTA-Student-TV reconstruction is:', min(error_FISTA_student_TV(:)));

figure(8); 
set(gcf, 'Position', get(0,'Screensize'));
subplot_tight(1,2,1, [0.05 0.05]); imshow(X_FISTA_student_TV,[0 0.6]); title('FISTA-Student-TV reconstruction'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); imshow((phantom - X_FISTA_student_TV).^2,[0 0.1]);  title('residual'); colorbar;
colormap(cmapnew); 

figure(9); 
subplot_tight(1,2,1, [0.05 0.05]); plot(error_FISTA_student_TV);  title('RMSE plot'); colorbar;
subplot_tight(1,2,2, [0.05 0.05]); plot(obj_FISTA_student_TV);  title('Objective plot'); colorbar;
colormap(cmapnew); 
%%
% print all RMSE's
fprintf('%s\n', '--------------------------------------------');
fprintf('%s %.4f\n', 'RMSE for FBP reconstruction:', RMSE(FBP_2(:), phantom(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-LS reconstruction:', min(error_FISTA(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-LS-TV reconstruction:', min(error_FISTA_TV(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-GH-TV reconstruction:', min(error_FISTA_GH_TV(:)));
fprintf('%s %.4f\n', 'Min RMSE for FISTA-Student-TV reconstruction:', min(error_FISTA_student_TV(:)));
%
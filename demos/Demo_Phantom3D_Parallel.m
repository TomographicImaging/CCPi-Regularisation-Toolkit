% A demo script to reconstruct 3D synthetic data using FISTA method for
% PARALLEL BEAM geometry
% requirements: ASTRA-toolbox and TomoPhantom toolbox

close all;clc;clear all;
% adding paths
addpath('../data/');
addpath('../main_func/'); addpath('../main_func/regularizers_CPU/'); addpath('../main_func/regularizers_GPU/NL_Regul/'); addpath('../main_func/regularizers_GPU/Diffus_HO/');
addpath('../supp/');

%%
% Main reconstruction/data generation parameters 
modelNo = 2; % see Phantom3DLibrary.dat file in TomoPhantom
N = 256; % x-y-z size (cubic image)
angles = 1:0.5:180; % angles vector in degrees
angles_rad = angles*(pi/180); % conversion to radians
det_size = round(sqrt(2)*N); % detector size

%---------TomoPhantom routines---------%
pathTP = '/home/algol/Documents/MATLAB/TomoPhantom/functions/models/Phantom3DLibrary.dat'; % path to TomoPhantom parameters file
TomoPhantom = buildPhantom3D(modelNo,N,pathTP); % generate 3D phantom
sino_tomophan3D = buildSino3D(modelNo, N, det_size, single(angles),pathTP); % generate ideal data
%--------------------------------------%
% Adding noise and distortions if required
sino_tomophan3D = sino_add_artifacts(sino_tomophan3D,'rings');
% adding Poisson noise
dose =  3e9; % photon flux (controls noise level)
multifactor = max(sino_tomophan3D(:));
dataExp = dose.*exp(-sino_tomophan3D/multifactor); % noiseless raw data
dataRaw = astra_add_noise_to_sino(dataExp, dose); % pre-log noisy raw data (weights)
sino3D_log = log(dose./max(dataRaw,1))*multifactor; %log corrected data -> sinogram
clear dataExp sino_tomophan3D
%
%%
% using ASTRA-toolbox to set the projection geometry (parallel beam)
proj_geom = astra_create_proj_geom('parallel', 1, det_size, angles_rad);
vol_geom = astra_create_vol_geom(N,N);
%%
fprintf('%s\n', 'Reconstructing with FBP using ASTRA-toolbox ...');
reconASTRA_3D = zeros(size(TomoPhantom),'single');
for k = 1:N
vol_id = astra_mex_data2d('create', '-vol', vol_geom, 0);
proj_id = astra_mex_data2d('create', '-sino', proj_geom, sino3D_log(:,:,k)'); 
cfg = astra_struct('FBP_CUDA');
cfg.ProjectionDataId = proj_id;
cfg.ReconstructionDataId = vol_id;
cfg.option.MinConstraint = 0;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 1);
rec = astra_mex_data2d('get', vol_id);
reconASTRA_3D(:,:,k) = single(rec);
end
figure; imshow(reconASTRA_3D(:,:,128), [0 1.3]);
%% 
%% 
fprintf('%s\n', 'Reconstruction using OS-FISTA-PWLS without regularization...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = single(sino3D_log); % sinogram
params.iterFISTA = 12; %max number of outer iterations
params.X_ideal = TomoPhantom; % ideal phantom
params.weights = dataRaw./max(dataRaw(:)); % statistical weight for PWLS
params.subsets = 12; % the number of subsets
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 1.3; 
tic; [X_FISTA, output] = FISTA_REC(params); toc; 

error_FISTA = output.Resid_error; obj_FISTA = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction is:', min(error_FISTA(:)));

Resid3D = (TomoPhantom - X_FISTA).^2;
figure(2);
subplot(1,2,1); imshow(X_FISTA(:,:,params.slice),[0 params.maxvalplot]); title('FISTA-LS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid3D(:,:,params.slice),[0 0.1]);  title('residual'); colorbar;
figure(3);
subplot(1,2,1); plot(error_FISTA);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA);  title('Objective plot'); 
%%
%% 
fprintf('%s\n', 'Reconstruction using OS-FISTA-GH without FGP-TV regularization...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = single(sino3D_log); % sinogram
params.iterFISTA = 15; %max number of outer iterations
params.X_ideal = TomoPhantom; % ideal phantom
params.weights = dataRaw./max(dataRaw(:)); % statistical weight for PWLS
params.subsets = 8; % the number of subsets
params.Regul_Lambda_FGPTV = 0.003; % TV regularization parameter for FGP-TV
params.Ring_LambdaR_L1 = 0.02; % Soft-Thresh L1 ring variable parameter
params.Ring_Alpha = 21; % to boost ring removal procedure
params.show = 1; % visualize reconstruction on each iteration
params.slice = 1; params.maxvalplot = 1.3; 
tic; [X_FISTA_GH_TV, output] = FISTA_REC(params); toc; 

error_FISTA_GH_TV = output.Resid_error; obj_FISTA_GH_TV = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for FISTA-PWLS reconstruction is:', min(error_FISTA_GH_TV(:)));

Resid3D = (TomoPhantom - X_FISTA_GH_TV).^2;
figure(2);
subplot(1,2,1); imshow(X_FISTA_GH_TV(:,:,params.slice),[0 params.maxvalplot]); title('FISTA-LS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid3D(:,:,params.slice),[0 0.1]);  title('residual'); colorbar;
figure(3);
subplot(1,2,1); plot(error_FISTA_GH_TV);  title('RMSE plot'); 
subplot(1,2,2); plot(obj_FISTA_GH_TV);  title('Objective plot'); 
%%
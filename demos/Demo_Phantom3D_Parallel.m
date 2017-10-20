% A demo script to reconstruct 3D synthetic data using FISTA method for
% PARALLEL BEAM geometry
% requirements: ASTRA-toolbox and TomoPhantom toolbox

close all;clc;clear all;
% adding paths
addpath('../data/');
addpath('../main_func/'); addpath('../main_func/regularizers_CPU/'); addpath('../main_func/regularizers_GPU/NL_Regul/'); addpath('../main_func/regularizers_GPU/Diffus_HO/');
addpath('../supp/');

%%
% build 3D phantom using TomoPhantom and generate projection data
modelNo = 2; % see Phantom3DLibrary.dat file in TomoPhantom
N = 256; % x-y-z size (cubic image)
angles = 1:0.5:180; % angles vector in degrees
angles_rad = angles*(pi/180); % conversion to radians
det_size = round(sqrt(2)*N); % detector size
% in order to run functions you have to go to the directory:
pathTP = '/home/algol/Documents/MATLAB/TomoPhantom/functions/models/Phantom3DLibrary.dat'; % path to TomoPhantom parameters file
TomoPhantom = buildPhantom3D(modelNo,N,pathTP); % generate 3D phantom
sino_tomophan3D = buildSino3D(modelNo, N, det_size, single(angles),pathTP); % generate ideal data
% Adding noise and distortions if required
sino_artifacts = sino_add_artifacts(sino_tomophan3D,'rings');
%
%%
% using ASTRA-toolbox to set the projection geometry (parallel beam)
proj_geom = astra_create_proj_geom('parallel', 1, det_size, angles_rad);
vol_geom = astra_create_vol_geom(N,N);
%%
fprintf('%s\n', 'Reconstructing with FBP using ASTRA-toolbox ...');
for i = 1:k
vol_id = astra_mex_data2d('create', '-vol', vol_geom, 0);
proj_id = astra_mex_data2d('create', '-proj3d', proj_geom, sino_artifacts(:,:,k)); 
cfg = astra_struct('FBP_CUDA');
cfg.ProjectionDataId = proj_id;
cfg.ReconstructionDataId = vol_id;
cfg.option.MinConstraint = 0;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 15);
reconASTRA_3D = astra_mex_data2d('get', vol_id);
end
%% 
fprintf('%s\n', 'Reconstruction using FISTA-LS without regularization...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = single(sino_tomophan3D); % sinogram
params.iterFISTA = 5; %max number of outer iterations
params.X_ideal = TomoPhantom; % ideal phantom
params.show = 1; % visualize reconstruction on each iteration
params.subsets = 12; 
params.slice = round(N/2); params.maxvalplot = 1; 
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
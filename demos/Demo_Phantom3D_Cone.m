% A demo script to reconstruct 3D synthetic data using FISTA method for
% CONE BEAM geometry
% requirements: ASTRA-toolbox and TomoPhantom toolbox

close all;clc;clear all;
% adding paths
addpath('../data/');
addpath('../main_func/'); addpath('../main_func/regularizers_CPU/'); addpath('../main_func/regularizers_GPU/NL_Regul/'); addpath('../main_func/regularizers_GPU/Diffus_HO/');
addpath('../supp/');


%%
% build 3D phantom using TomoPhantom 
modelNo = 3; % see Phantom3DLibrary.dat file in TomoPhantom
N = 256; % x-y-z size (cubic image)
angles = 0:1.5:360; % angles vector in degrees
angles_rad = angles*(pi/180); % conversion to radians
det_size = round(sqrt(2)*N); % detector size
% in order to run functions you have to go to the directory:
pathTP = '/home/algol/Documents/MATLAB/TomoPhantom/functions/models/Phantom3DLibrary.dat'; % path to TomoPhantom parameters file
TomoPhantom = buildPhantom3D(modelNo,N,pathTP); % generate 3D phantom
%%
% using ASTRA-toolbox to set the projection geometry (cone beam)
% eg: astra.create_proj_geom('cone', 1.0 (resol), 1.0 (resol), detectorRowCount, detectorColCount, angles, originToSource, originToDetector)
vol_geom = astra_create_vol_geom(N,N,N);
proj_geom = astra_create_proj_geom('cone', 1.0, 1.0, N, det_size, angles_rad, 2000, 2160);
%% 
% do forward projection using ASTRA
% inverse crime data generation
[sino_id, SinoCone3D] = astra_create_sino3d_cuda(TomoPhantom, proj_geom, vol_geom);
astra_mex_data3d('delete', sino_id);
%%
fprintf('%s\n', 'Reconstructing with CGLS using ASTRA-toolbox ...');
vol_id = astra_mex_data3d('create', '-vol', vol_geom, 0);
proj_id = astra_mex_data3d('create', '-proj3d', proj_geom, SinoCone3D); 
cfg = astra_struct('CGLS3D_CUDA');
cfg.ProjectionDataId = proj_id;
cfg.ReconstructionDataId = vol_id;
cfg.option.MinConstraint = 0;
alg_id = astra_mex_algorithm('create', cfg);
astra_mex_algorithm('iterate', alg_id, 15);
reconASTRA_3D = astra_mex_data3d('get', vol_id);
%%
fprintf('%s\n', 'Reconstruction using FISTA-LS without regularization...');
clear params
% define parameters
params.proj_geom = proj_geom; % pass geometry to the function
params.vol_geom = vol_geom;
params.sino = single(SinoCone3D); % sinogram
params.iterFISTA = 30; %max number of outer iterations
params.X_ideal = TomoPhantom; % ideal phantom
params.show = 1; % visualize reconstruction on each iteration
params.slice = round(N/2); params.maxvalplot = 1; 
tic; [X_FISTA, output] = FISTA_REC(params); toc; 

error_FISTA = output.Resid_error; obj_FISTA = output.objective;
fprintf('%s %.4f\n', 'Min RMSE for FISTA-LS reconstruction is:', min(error_FISTA(:)));

Resid3D = (TomoPhantom - X_FISTA).^2;
figure(2);
subplot(1,2,1); imshow(X_FISTA(:,:,params.slice),[0 params.maxvalplot]); title('FISTA-LS reconstruction'); colorbar;
subplot(1,2,2); imshow(Resid3D(:,:,params.slice),[0 0.1]);  title('residual'); colorbar;
figure(3);
subplot(1,2,1); plot(error_FISTA);  title('RMSE plot'); colorbar;
subplot(1,2,2); plot(obj_FISTA);  title('Objective plot'); colorbar;
%%
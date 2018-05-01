% Image (2D) denoising demo using CCPi-RGL
clear; close all
Path1 = sprintf(['..' filesep 'mex_compile' filesep 'installed'], 1i);
Path2 = sprintf(['..' filesep '..' filesep '..' filesep 'data' filesep], 1i);
addpath(Path1);
addpath(Path2);

Im = double(imread('lena_gray_512.tif'))/255;  % loading image
u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
figure; imshow(u0, [0 1]); title('Noisy image');

lambda_reg = 0.03; % regularsation parameter for all methods
%%
fprintf('Denoise using the ROF-TV model (CPU) \n');
tau_rof = 0.0025; % time-marching constant 
iter_rof = 750; % number of ROF iterations
tic; u_rof = ROF_TV(single(u0), lambda_reg, iter_rof, tau_rof); toc; 
energyfunc_val_rof = TV_energy(single(u_rof),single(u0),lambda_reg, 1);  % get energy function value
figure; imshow(u_rof, [0 1]); title('ROF-TV denoised image (CPU)');
%%
% fprintf('Denoise using the ROF-TV model (GPU) \n');
% tau_rof = 0.0025; % time-marching constant 
% iter_rof = 750; % number of ROF iterations
% tic; u_rofG = ROF_TV_GPU(single(u0), lambda_reg, iter_rof, tau_rof); toc;
% figure; imshow(u_rofG, [0 1]); title('ROF-TV denoised image (GPU)');
%%
fprintf('Denoise using the FGP-TV model (CPU) \n');
iter_fgp = 1000; % number of FGP iterations
epsil_tol =  1.0e-06; % tolerance
tic; u_fgp = FGP_TV(single(u0), lambda_reg, iter_fgp, epsil_tol); toc; 
energyfunc_val_fgp = TV_energy(single(u_fgp),single(u0),lambda_reg, 1); % get energy function value
figure; imshow(u_fgp, [0 1]); title('FGP-TV denoised image (CPU)');

%%
% fprintf('Denoise using the FGP-TV model (GPU) \n');
% iter_fgp = 1000; % number of FGP iterations
% epsil_tol =  1.0e-05; % tolerance
% tic; u_fgpG = FGP_TV_GPU(single(u0), lambda_reg, iter_fgp, epsil_tol); toc; 
% figure; imshow(u_fgpG, [0 1]); title('FGP-TV denoised image (GPU)');
%%
fprintf('Denoise using the SB-TV model (CPU) \n');
iter_sb = 150; % number of SB iterations
epsil_tol =  1.0e-06; % tolerance
tic; u_sb = SB_TV(single(u0), lambda_reg, iter_sb, epsil_tol); toc; 
energyfunc_val_sb = TV_energy(single(u_sb),single(u0),lambda_reg, 1);  % get energy function value
figure; imshow(u_sb, [0 1]); title('SB-TV denoised image (CPU)');
%%
% fprintf('Denoise using the SB-TV model (GPU) \n');
% iter_sb = 150; % number of SB iterations
% epsil_tol =  1.0e-06; % tolerance
% tic; u_sbG = SB_TV_GPU(single(u0), lambda_reg, iter_sb, epsil_tol); toc; 
% figure; imshow(u_sbG, [0 1]); title('SB-TV denoised image (GPU)');
%%
fprintf('Denoise using Nonlinear-Diffusion model (CPU) \n');
iter_diff = 800; % number of diffusion iterations
lambda_regDiff = 0.06; % regularisation for the diffusivity 
sigmaPar = 0.04; % edge-preserving parameter
tau_param = 0.025; % time-marching constant 
tic; u_diff = NonlDiff(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber'); toc; 
figure; imshow(u_diff, [0 1]); title('Diffusion denoised image (CPU)');
%%
% fprintf('Denoise using Nonlinear-Diffusion model (GPU) \n');
% iter_diff = 800; % number of diffusion iterations
% lambda_regDiff = 0.06; % regularisation for the diffusivity 
% sigmaPar = 0.04; % edge-preserving parameter
% tau_param = 0.025; % time-marching constant 
% tic; u_diff_g = NonlDiff_GPU(single(u0), lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber'); toc; 
% figure; imshow(u_diff_g, [0 1]); title('Diffusion denoised image (GPU)');
%%
%>>>>>>>>>>>>>> MULTI-CHANNEL priors <<<<<<<<<<<<<<< %

fprintf('Denoise using the FGP-dTV model (CPU) \n');
% create another image (reference) with slightly less amount of noise
u_ref = Im + .01*randn(size(Im)); u_ref(u_ref < 0) = 0;
% u_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)

iter_fgp = 1000; % number of FGP iterations
epsil_tol =  1.0e-06; % tolerance
eta =  0.2; % Reference image gradient smoothing constant
tic; u_fgp_dtv = FGP_dTV(single(u0), single(u_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
figure; imshow(u_fgp_dtv, [0 1]); title('FGP-dTV denoised image (CPU)');
%%
% fprintf('Denoise using the FGP-dTV model (GPU) \n');
% % create another image (reference) with slightly less amount of noise
% u_ref = Im + .01*randn(size(Im)); u_ref(u_ref < 0) = 0;
% % u_ref = zeros(size(Im),'single'); % pass zero reference (dTV -> TV)
% 
% iter_fgp = 1000; % number of FGP iterations
% epsil_tol =  1.0e-06; % tolerance
% eta =  0.2; % Reference image gradient smoothing constant
% tic; u_fgp_dtvG = FGP_dTV_GPU(single(u0), single(u_ref), lambda_reg, iter_fgp, epsil_tol, eta); toc; 
% figure; imshow(u_fgp_dtvG, [0 1]); title('FGP-dTV denoised image (GPU)');
%%
fprintf('Denoise using the TNV prior (CPU) \n');
slices = 5; N = 512;
vol3D = zeros(N,N,slices, 'single');
for i = 1:slices
vol3D(:,:,i) = Im + .05*randn(size(Im)); 
end
vol3D(vol3D < 0) = 0;

iter_tnv = 200; % number of TNV iterations
tic; u_tnv = TNV(single(vol3D), lambda_reg, iter_tnv); toc; 
figure; imshow(u_tnv(:,:,3), [0 1]); title('TNV denoised stack of channels (CPU)');

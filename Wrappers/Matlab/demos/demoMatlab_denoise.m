% Image (2D) denoising demo using CCPi-RGL

addpath('../mex_compile/installed');
addpath('../../../data/');

Im = double(imread('lena_gray_512.tif'))/255;  % loading image
u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
figure; imshow(u0, [0 1]); title('Noisy image');

%%
fprintf('Denoise using ROF-TV model (CPU) \n');
lambda_rof = 0.03; % regularisation parameter
tau_rof = 0.0025; % time-marching constant 
iter_rof = 2000; % number of ROF iterations
tic; u_rof = ROF_TV(single(u0), lambda_rof, iter_rof, tau_rof); toc; 
figure; imshow(u_rof, [0 1]); title('ROF-TV denoised image (CPU)');
%%
% fprintf('Denoise using ROF-TV model (GPU) \n');
% lambda_rof = 0.03; % regularisation parameter
% tau_rof = 0.0025; % time-marching constant 
% iter_rof = 2000; % number of ROF iterations
% tic; u_rofG = ROF_TV_GPU(single(u0), lambda_rof, iter_rof, tau_rof); toc;
% figure; imshow(u_rofG, [0 1]); title('ROF-TV denoised image (GPU)');
%%
fprintf('Denoise using FGP-TV model (CPU) \n');
lambda_fgp = 0.03; % regularisation parameter
iter_fgp = 1000; % number of FGP iterations
epsil_tol =  1.0e-05; % tolerance
tic; u_fgp = FGP_TV(single(u0), lambda_fgp, iter_fgp, epsil_tol); toc; 
figure; imshow(u_fgp, [0 1]); title('FGP-TV denoised image (CPU)');
%%
% fprintf('Denoise using FGP-TV model (GPU) \n');
% lambda_fgp = 0.03; % regularisation parameter
% iter_fgp = 1000; % number of FGP iterations
% epsil_tol =  1.0e-05; % tolerance
% tic; u_fgpG = FGP_TV_GPU(single(u0), lambda_fgp, iter_fgp, epsil_tol); toc; 
% figure; imshow(u_fgpG, [0 1]); title('FGP-TV denoised image (GPU)');
%%

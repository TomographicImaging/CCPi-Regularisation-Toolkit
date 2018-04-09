% Volume (3D) denoising demo using CCPi-RGL

addpath('../mex_compile/installed');
addpath('../../../data/');

N = 256; 
slices = 30;
vol3D = zeros(N,N,slices, 'single');
Im = double(imread('lena_gray_256.tif'))/255;  % loading image
for i = 1:slices
vol3D(:,:,i) = Im + .05*randn(size(Im)); 
end
vol3D(vol3D < 0) = 0;
figure; imshow(vol3D(:,:,15), [0 1]); title('Noisy image');

%%
fprintf('Denoise using ROF-TV model (CPU) \n');
lambda_rof = 0.03; % regularisation parameter
tau_rof = 0.0025; % time-marching constant 
iter_rof = 1000; % number of ROF iterations
tic; u_rof = ROF_TV(single(vol3D), lambda_rof, iter_rof, tau_rof); toc; 
figure; imshow(u_rof(:,:,15), [0 1]); title('ROF-TV denoised volume (CPU)');
%%
% fprintf('Denoise using ROF-TV model (GPU) \n');
% lambda_rof = 0.03; % regularisation parameter
% tau_rof = 0.0025; % time-marching constant 
% iter_rof = 1000; % number of ROF iterations
% tic; u_rofG = ROF_TV_GPU(single(vol3D), lambda_rof, iter_rof, tau_rof); toc;
% figure; imshow(u_rofG(:,:,15), [0 1]); title('ROF-TV denoised volume (GPU)');
%%
fprintf('Denoise using FGP-TV model (CPU) \n');
lambda_fgp = 0.03; % regularisation parameter
iter_fgp = 500; % number of FGP iterations
epsil_tol =  1.0e-05; % tolerance
tic; u_fgp = FGP_TV(single(vol3D), lambda_fgp, iter_fgp, epsil_tol); toc; 
figure; imshow(u_fgp(:,:,15), [0 1]); title('FGP-TV denoised volume (CPU)');
%%
% fprintf('Denoise using FGP-TV model (GPU) \n');
% lambda_fgp = 0.03; % regularisation parameter
% iter_fgp = 500; % number of FGP iterations
% epsil_tol =  1.0e-05; % tolerance
% tic; u_fgpG = FGP_TV_GPU(single(vol3D), lambda_fgp, iter_fgp, epsil_tol); toc; 
% figure; imshow(u_fgpG(:,:,15), [0 1]); title('FGP-TV denoised volume (GPU)');
%%
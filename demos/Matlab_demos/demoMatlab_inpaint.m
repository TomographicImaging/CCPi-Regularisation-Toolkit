% Image (2D) inpainting demo using CCPi-RGL
clear; close all

fsep = '/';

Path1 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'mex_compile' fsep 'installed'], 1i);
Path2 = sprintf(['..' fsep 'data' fsep], 1i);
Path3 = sprintf(['..' fsep '..' fsep 'src' fsep 'Matlab' fsep 'supp'], 1i);
addpath(Path1);
addpath(Path2);
addpath(Path3);

load('SinoInpaint.mat');
Sinogram = Sinogram./max(Sinogram(:));
Sino_mask = Sinogram.*(1-single(Mask));
figure; 
subplot(1,2,1); imshow(Sino_mask, [0 1]); title('Missing data sinogram');
subplot(1,2,2); imshow(Mask, [0 1]); title('Mask');
%%
fprintf('Inpaint using Linear-Diffusion model (CPU) \n');
iter_diff = 5000; % number of diffusion iterations
lambda_regDiff = 6000; % regularisation for the diffusivity 
sigmaPar = 0.0; % edge-preserving parameter
tau_param = 0.000075; % time-marching constant 
tic; u_diff = NonlDiff_Inp(single(Sino_mask), Mask, lambda_regDiff, sigmaPar, iter_diff, tau_param); toc; 
figure; imshow(u_diff, [0 1]); title('Linear-Diffusion inpainted sinogram (CPU)');
%%
fprintf('Inpaint using Nonlinear-Diffusion model (CPU) \n');
iter_diff = 1500; % number of diffusion iterations
lambda_regDiff = 80; % regularisation for the diffusivity 
sigmaPar = 0.00009; % edge-preserving parameter
tau_param = 0.000008; % time-marching constant 
tic; u_diff = NonlDiff_Inp(single(Sino_mask), Mask, lambda_regDiff, sigmaPar, iter_diff, tau_param, 'Huber'); toc; 
figure; imshow(u_diff, [0 1]); title('Non-Linear Diffusion inpainted sinogram (CPU)');
%%
fprintf('Inpaint using Nonlocal Vertical Marching model (CPU) \n');
Increment = 1; % linear increment for the searching window
tic; [u_nom,maskupd] = NonlocalMarching_Inpaint(single(Sino_mask), Mask, Increment); toc;
figure; imshow(u_nom, [0 1]); title('NVM inpainted sinogram (CPU)');
%%
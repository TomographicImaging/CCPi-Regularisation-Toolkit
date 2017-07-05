function [X,  output] = FISTA_REC(params)

% <<<< FISTA-based reconstruction algorithm using ASTRA-toolbox >>>>
% ___Input___:
% params.[] file:
%       - .proj_geom (geometry of the projector) [required]
%       - .vol_geom (geometry of the reconstructed object) [required]
%       - .sino (vectorized in 2D or 3D sinogram) [required]
%       - .iterFISTA (iterations for the main loop, default 40)
%       - .L_const (Lipschitz constant, default Power method)                                                                                                    )
%       - .X_ideal (ideal image, if given)
%       - .weights (statisitcal weights, size of the sinogram)
%       - .ROI (Region-of-interest, only if X_ideal is given)
%       - .initialize (a 'warm start' using SIRT method from ASTRA)
%----------------Regularization choices------------------------
%       - .Regul_Lambda_FGPTV (FGP-TV regularization parameter)
%       - .Regul_Lambda_SBTV (SplitBregman-TV regularization parameter)
%       - .Regul_Lambda_L1 (L1 regularization by soft-thresholding)
%       - .Regul_Lambda_TVLLT (Higher order SB-LLT regularization parameter)
%       - .Regul_tol (tolerance to terminate regul iterations, default 1.0e-04)
%       - .Regul_Iterations (iterations for the selected penalty, default 25)
%       - .Regul_tauLLT (time step parameter for LLT term)
%       - .Ring_LambdaR_L1 (regularization parameter for L1-ring minimization, if lambdaR_L1 > 0 then switch on ring removal)
%       - .Ring_Alpha (larger values can accelerate convergence but check stability, default 1)
%----------------Visualization parameters------------------------
%       - .show (visualize reconstruction 1/0, (0 default))
%       - .maxvalplot (maximum value to use for imshow[0 maxvalplot])
%       - .slice (for 3D volumes - slice number to imshow)
% ___Output___:
% 1. X - reconstructed image/volume
% 2. output - structure with
%    - .Resid_error - residual error (if X_ideal is given)
%    - .objective: value of the objective function
%    - .L_const: Lipshitz constant to avoid recalculations

% References:
% 1. "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
% Problems" by A. Beck and M Teboulle
% 2. "Ring artifacts correction in compressed sensing..." by P. Paleo
% 3. "A novel tomographic reconstruction method based on the robust
% Student's t function for suppressing data outliers" D. Kazantsev et.al.
% D. Kazantsev, 2016-17

% Dealing with input parameters
if (isfield(params,'proj_geom') == 0)
    error('%s \n', 'Please provide ASTRA projection geometry - proj_geom');
else
    proj_geom = params.proj_geom;
end
if (isfield(params,'vol_geom') == 0)
    error('%s \n', 'Please provide ASTRA object geometry - vol_geom');
else
    vol_geom = params.vol_geom;
end
N = params.vol_geom.GridColCount;
if (isfield(params,'sino'))
    sino = params.sino;
    [Detectors, anglesNumb, SlicesZ] = size(sino);
    fprintf('%s %i %s %i %s %i %s \n', 'Sinogram has a dimension of', Detectors, 'detectors;', anglesNumb, 'projections;', SlicesZ, 'vertical slices.');
else
    error('%s \n', 'Please provide a sinogram');
end
if (isfield(params,'iterFISTA'))
    iterFISTA = params.iterFISTA;
else
    iterFISTA = 40;
end
if (isfield(params,'weights'))
    weights = params.weights;
else
    weights = ones(size(sino));
end
if (isfield(params,'L_const'))
    L_const = params.L_const;
else
    % using Power method (PM) to establish L constant
    niter = 8; % number of iteration for PM
    x = rand(N,N,SlicesZ);
    sqweight = sqrt(weights);
    [sino_id, y] = astra_create_sino3d_cuda(x, proj_geom, vol_geom);
    y = sqweight.*y;
    astra_mex_data3d('delete', sino_id);
    
    for i = 1:niter
        [id,x] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geom, vol_geom);
        s = norm(x(:));
        x = x/s;
        [sino_id, y] = astra_create_sino3d_cuda(x, proj_geom, vol_geom);
        y = sqweight.*y;
        astra_mex_data3d('delete', sino_id);
        astra_mex_data3d('delete', id);
    end
    L_const = s;
end
if (isfield(params,'X_ideal'))
    X_ideal = params.X_ideal;
else
    X_ideal = 'none';
end
if (isfield(params,'ROI'))
    ROI = params.ROI;
else
    ROI = find(X_ideal>=0.0);
end
if (isfield(params,'Regul_Lambda_FGPTV'))
    lambdaFGP_TV = params.Regul_Lambda_FGPTV;
else
    lambdaFGP_TV = 0;
end
if (isfield(params,'Regul_Lambda_SBTV'))
    lambdaSB_TV = params.Regul_Lambda_SBTV;
else
    lambdaSB_TV = 0;
end
if (isfield(params,'Regul_Lambda_L1'))
    lambdaL1 = params.Regul_Lambda_L1;
else
    lambdaL1 = 0;
end
if (isfield(params,'Regul_tol'))
    tol = params.Regul_tol;
else
    tol = 1.0e-04;
end
if (isfield(params,'Regul_Iterations'))
    IterationsRegul = params.Regul_Iterations;
else
    IterationsRegul = 25;
end
if (isfield(params,'Regul_LambdaHO'))
    lambdaHO = params.Regul_LambdaHO;
else
    lambdaHO = 0;
end
if (isfield(params,'Regul_iterHO'))
    iterHO = params.Regul_iterHO;
else
    iterHO = 50;
end
if (isfield(params,'Regul_tauLLT'))
    tauHO = params.Regul_tauLLT;
else
    tauHO = 0.0001;
end
if (isfield(params,'Ring_LambdaR_L1'))
    lambdaR_L1 = params.Ring_LambdaR_L1;
else
    lambdaR_L1 = 0;
end
if (isfield(params,'Ring_Alpha'))
    alpha_ring = params.Ring_Alpha; % higher values can accelerate ring removal procedure
else
    alpha_ring = 1;
end
if (isfield(params,'show'))
    show = params.show;
else
    show = 0;
end
if (isfield(params,'maxvalplot'))
    maxvalplot = params.maxvalplot;
else
    maxvalplot = 1;
end
if (isfield(params,'slice'))
    slice = params.slice;
else
    slice = 1;
end
if (isfield(params,'initialize'))
    % a 'warm start' with SIRT method
    % Create a data object for the reconstruction
    rec_id = astra_mex_data3d('create', '-vol', vol_geom);
    
    sinogram_id = astra_mex_data3d('create', '-proj3d', proj_geom, sino);
    
    % Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra_struct('SIRT3D_CUDA');
    cfg.ReconstructionDataId = rec_id;
    cfg.ProjectionDataId = sinogram_id;
    
    % Create the algorithm object from the configuration structure
    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('iterate', alg_id, 35);
    % Get the result
    X = astra_mex_data3d('get', rec_id);
    
    % Clean up. Note that GPU memory is tied up in the algorithm object,
    % and main RAM in the data objects.
    astra_mex_algorithm('delete', alg_id);
    astra_mex_data3d('delete', rec_id);
    astra_mex_data3d('delete', sinogram_id);
else
    X = zeros(N,N,SlicesZ, 'single'); % storage for the solution
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resid_error = zeros(iterFISTA,1); % error vector
objective = zeros(iterFISTA,1); % obhective vector

t = 1;
X_t = X;

% add_ring = zeros(size(sino),'single'); % size of sinogram array
r = zeros(Detectors,SlicesZ, 'single'); % 2D array (for 3D data) of sparse "ring" vectors
r_x = r; % another ring variable
residual = zeros(size(sino),'single');

% Outer iterations loop
for i = 1:iterFISTA
    
    X_old = X;
    t_old = t;
    r_old = r;
    
    [sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geom, vol_geom);
    
    if (lambdaR_L1 > 0)
        % add ring removal part (Group-Huber fidelity)
        for kkk = 1:anglesNumb
            % add_ring(:,kkk,:) =  squeeze(sino(:,kkk,:)) - alpha_ring.*r_x;
            residual(:,kkk,:) =  squeeze(weights(:,kkk,:)).*(squeeze(sino_updt(:,kkk,:)) - (squeeze(sino(:,kkk,:)) - alpha_ring.*r_x));
        end
        
        vec = sum(residual,2);
        if (SlicesZ > 1)
            vec = squeeze(vec(:,1,:));
        end
        r = r_x - (1./L_const).*vec;
    else
        % no ring removal
        residual = weights.*(sino_updt - sino);
    end
    % residual =  weights.*(sino_updt - add_ring);
    
    [id, x_temp] = astra_create_backprojection3d_cuda(residual, proj_geom, vol_geom);
    
    X = X_t - (1/L_const).*x_temp;
    astra_mex_data3d('delete', sino_id);
    astra_mex_data3d('delete', id);
    
    if (lambdaFGP_TV > 0)
        % FGP-TV regularization
        [X, f_val] = FGP_TV(single(X), lambdaFGP_TV, IterationsRegul, tol, 'iso');
        objective(i) = 0.5.*norm(residual(:))^2 + f_val;
    end
    if (lambdaSB_TV > 0)
        % Split Bregman regularization
        X = SplitBregman_TV(single(X), lambdaSB_TV, IterationsRegul, tol);  % (more memory efficent)
        objective(i) = 0.5.*norm(residual(:))^2;
    end
    if (lambdaL1 > 0)
        % L1 soft-threhsolding regularization
        X = max(abs(X)-lambdaL1, 0).*sign(X);
        objective(i) = 0.5.*norm(residual(:))^2;
    end
    if (lambdaHO > 0)
        % Higher Order (LLT) regularization
        X2 = LLT_model(single(X), lambdaHO, tauHO, iterHO, 3.0e-05, 0);
        X = 0.5.*(X + X2); % averaged combination of two solutions
        objective(i) = 0.5.*norm(residual(:))^2;
    end
    
    if (lambdaR_L1 > 0)
        r =  max(abs(r)-lambdaR_L1, 0).*sign(r); % soft-thresholding operator
    end
    
    t = (1 + sqrt(1 + 4*t^2))/2; % updating t
    X_t = X + ((t_old-1)/t).*(X - X_old); % updating X
    
    if (lambdaR_L1 > 0)
        r_x = r + ((t_old-1)/t).*(r - r_old); % updating r
    end
    
    if (show == 1)
        figure(10); imshow(X(:,:,slice), [0 maxvalplot]);
        if (lambdaR_L1 > 0)
            figure(11); plot(r); title('Rings offset vector')
        end
        pause(0.01);
    end
    if (strcmp(X_ideal, 'none' ) == 0)
        Resid_error(i) = RMSE(X(ROI), X_ideal(ROI));
        fprintf('%s %i %s %s %.4f  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Error RMSE:', Resid_error(i), '|', 'Objective:', objective(i));
    else
        fprintf('%s %i  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Objective:', objective(i));
    end
end
output.Resid_error = Resid_error;
output.objective = objective;
output.L_const = L_const;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

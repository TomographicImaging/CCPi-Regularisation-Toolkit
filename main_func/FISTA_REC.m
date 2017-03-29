function [X,  error, objective, residual] = FISTA_REC(params)

% <<<< FISTA-based reconstruction algorithm using ASTRA-toolbox (parallel beam) >>>>
% ___Input___:
% params.[] file:
%       - .sino (2D or 3D sinogram) [required]
%       - .N (image dimension) [required]
%       - .angles (in radians) [required]
%       - .iterFISTA (iterations for the main loop)
%       - .L_const (Lipschitz constant, default Power method)                                                                                                    )
%       - .X_ideal (ideal image, if given)
%       - .weights (statisitcal weights, size of sinogram)
%       - .ROI (Region-of-interest, only if X_ideal is given)
%       - .lambdaTV (TV regularization parameter, default 0 - reg. TV is switched off)
%       - .tol (tolerance to terminate TV regularization, default 1.0e-04)
%       - .iterTV (iterations for the TV penalty, default 0)
%       - .lambdaHO (Higher Order LLT regularization parameter, default 0 - LLT reg. switched off)
%       - .iterHO (iterations for HO penalty, default 50)
%       - .tauHO (time step parameter for HO term)
%       - .lambdaR_L1 (regularization parameter for L1 ring minimization, if lambdaR_L1 > 0 then switch on ring removal, default 0)
%       - .alpha_ring (larger values can accelerate convergence but check stability, default 1)
%       - .fidelity (choose between "LS" and "student" data fidelities)
%       - .precondition (1 - switch on Fourier filtering before backprojection)
%       - .show (visualize reconstruction 1/0, (0 default))
%       - .maxvalplot (maximum value to use for imshow[0 maxvalplot])
%       - .slice (for 3D volumes - slice number to imshow)
% ___Output___:
% 1. X - reconstructed image/volume
% 2. error - residual error (if X_ideal is given)
% 3. value of the objective function
% 4. forward projection(X)
% References:
% 1. "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
% Problems" by A. Beck and M Teboulle
% 2. "Ring artifacts correction in compressed sensing..." by P. Paleo
% 3. "A novel tomographic reconstruction method based on the robust
% Student's t function for suppressing data outliers" D. Kazantsev et.al.
% D. Kazantsev, 2016-17

% Dealing with input parameters
if (isfield(params,'sino'))
    sino = params.sino;
    [anglesNumb, Detectors, SlicesZ] = size(sino);
    fprintf('%s %i %s %i %s %i %s \n', 'Sinogram has a dimension of', anglesNumb, 'projections;', Detectors, 'detectors;', SlicesZ, 'vertical slices.');
else
    fprintf('%s \n', 'Please provide a sinogram');
end
if (isfield(params,'N'))
    N = params.N;
else
    fprintf('%s \n', 'Please provide N-size for the reconstructed image [N x N]');
end
if (isfield(params,'N'))
    angles = params.angles;
    if (length(angles) ~= anglesNumb)
        fprintf('%s \n', 'Sinogram angular dimension does not correspond to the angles dimension provided');
    end
else
    fprintf('%s \n', 'Please provide a vector of angles');
end
if (isfield(params,'iterFISTA'))
    iterFISTA = params.iterFISTA;
else
    iterFISTA = 30;
end
if (isfield(params,'L_const'))
    L_const = params.L_const;
else
    % using Power method (PM) to establish L constant
    vol_geom = astra_create_vol_geom(N, N);
    proj_geom = astra_create_proj_geom('parallel', 1.0, Detectors, angles);
    
    niter = 10; % number of iteration for PM
    x = rand(N,N);
    [sino_id, y] = astra_create_sino_cuda(x, proj_geom, vol_geom);
    astra_mex_data2d('delete', sino_id);
    
    for i = 1:niter
        x = astra_create_backprojection_cuda(y, proj_geom, vol_geom);
        s = norm(x);
        x = x/s;
        [sino_id, y] = astra_create_sino_cuda(x, proj_geom, vol_geom);
        astra_mex_data2d('delete', sino_id);
    end
    L_const = s;
end
if (isfield(params,'X_ideal'))
    X_ideal = params.X_ideal;
else
    X_ideal = 'none';
end
if (isfield(params,'weights'))
    weights = params.weights;
else
    weights = 1;
end
if (isfield(params,'ROI'))
    ROI = params.ROI;
else
    ROI = find(X_ideal>=0.0);
end
if (isfield(params,'lambdaTV'))
    lambdaTV = params.lambdaTV;
else
    lambdaTV = 0;
end
if (isfield(params,'tol'))
    tol = params.tol;
else
    tol = 1.0e-04;
end
if (isfield(params,'iterTV'))
    iterTV = params.iterTV;
else
    iterTV = 10;
end
if (isfield(params,'lambdaHO'))
    lambdaHO = params.lambdaHO;
else
    lambdaHO = 0;
end
if (isfield(params,'iterHO'))
    iterHO = params.iterHO;
else
    iterHO = 50;
end
if (isfield(params,'tauHO'))
    tauHO = params.tauHO;
else
    tauHO = 0.0001;
end
if (isfield(params,'lambdaR_L1'))
    lambdaR_L1 = params.lambdaR_L1;    
else
    lambdaR_L1 = 0;
end
if (isfield(params,'alpha_ring'))    
    alpha_ring = params.alpha_ring; % higher values can accelerate ring removal procedure
else
    alpha_ring = 1;
end
if (isfield(params,'fidelity'))
    fidelity = params.fidelity;
else
    fidelity = 'LS';
end
if (isfield(params,'precondition'))
    precondition = params.precondition;
else
    precondition = 0;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% building geometry (parallel-beam)
vol_geom = astra_create_vol_geom(N, N);
proj_geom = astra_create_proj_geom('parallel', 1.0, Detectors, angles);
error = zeros(iterFISTA,1); % error vector
objective = zeros(iterFISTA,1); % obhective vector

if (lambdaR_L1 > 0)
    % do reconstruction WITH ring removal (Group-Huber fidelity)
    t = 1;
    X = zeros(N,N,SlicesZ, 'single');
    X_t = X;
    
    add_ring = zeros(anglesNumb, Detectors, SlicesZ, 'single'); % size of sinogram array
    r = zeros(Detectors,SlicesZ, 'single'); % 2D array (for 3D data) of sparse "ring" vectors
    r_x = r;
    
    % iterations loop
    for i = 1:iterFISTA
        
        X_old = X;
        t_old = t;
        r_old = r;
        
        % all slices loop
        for j = 1:SlicesZ
            
            [sino_id, sino_updt] = astra_create_sino_cuda(X_t(:,:,j), proj_geom, vol_geom);
            
            for kkk = 1:anglesNumb
                add_ring(kkk,:,j) =  sino(kkk,:,j) - alpha_ring.*r_x(:,j)';
            end
            
            residual =  sino_updt - add_ring(:,:,j);
            
            if (precondition == 1)
                residual = filtersinc(residual'); % filtering residual (Fourier preconditioning)
                residual = residual';
            end
            
            vec = sum(residual);
            r(:,j) = r_x(:,j) - (1/L_const).*vec';
            
            x_temp = astra_create_backprojection_cuda(residual, proj_geom, vol_geom);
            
            X(:,:,j) = X_t(:,:,j) - (1/L_const).*x_temp;
            astra_mex_data2d('delete', sino_id);
        end
        
        if ((lambdaTV > 0) && (lambdaHO == 0))
            if (size(X,3) > 1) 
                [X] = FISTA_TV(single(X), lambdaTV, iterTV, tol); % TV regularization using FISTA
                 gradTV = 1;
            else
                [X, gradTV] = FISTA_TV(single(X), lambdaTV, iterTV, tol); % TV regularization using FISTA
            end
            objective(i) = 0.5.*norm(residual(:))^2 + norm(gradTV(:));
            %       X = SplitBregman_TV(single(X), lambdaTV, iterTV, tol);  % TV-Split Bregman regularization on CPU (memory limited)
        elseif ((lambdaHO > 0) && (lambdaTV == 0))
            % Higher Order regularization
            X = LLT_model(single(X), lambdaHO, tauHO, iterHO, tol, 0);   % LLT higher order model
        elseif ((lambdaTV > 0) && (lambdaHO > 0))
            %X1 = SplitBregman_TV(single(X), lambdaTV, iterTV, tol);     % TV-Split Bregman regularization on CPU (memory limited)
            X1 = FISTA_TV(single(X), lambdaTV, iterTV, tol); % TV regularization using FISTA
            X2 = LLT_model(single(X), lambdaHO, tauHO, iterHO, tol, 0);   % LLT higher order model
            X = 0.5.*(X1 + X2); % averaged combination of two solutions
        elseif ((lambdaTV == 0) && (lambdaHO == 0))
            objective(i) = 0.5.*norm(residual(:))^2;
        end
        
        r =  max(abs(r)-lambdaR_L1, 0).*sign(r); % soft-thresholding operator
        
        t = (1 + sqrt(1 + 4*t^2))/2; % updating t
        X_t = X + ((t_old-1)/t).*(X - X_old); % updating X
        r_x = r + ((t_old-1)/t).*(r - r_old); % updating r
        
        if (show == 1)
            figure(10); imshow(X(:,:,slice), [0 maxvalplot]);
            figure(11); plot(r); title('Rings offset vector')
            pause(0.03);            
        end
        if (strcmp(X_ideal, 'none' ) == 0)
            error(i) = RMSE(X(ROI), X_ideal(ROI));
            fprintf('%s %i %s %s %.4f  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Error RMSE:', error(i), '|', 'Objective:', objective(i));
        else
            fprintf('%s %i  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Objective:', objective(i));
        end
        
    end
    
else
    % WITHOUT ring removal
    t = 1;
    X = zeros(N,N,SlicesZ, 'single');
    X_t = X;
    
    % iterations loop
    for i = 1:iterFISTA
        
        X_old = X;
        t_old = t;
        
        % slices loop
        for j = 1:SlicesZ
            [sino_id, sino_updt] = astra_create_sino_cuda(X_t(:,:,j), proj_geom, vol_geom);
            residual = weights.*(sino_updt - sino(:,:,j));
            
            % employ students t fidelity term
            if (strcmp(fidelity,'student') == 1)
                res_vec = reshape(residual, anglesNumb*Detectors,1);
                %s = 100;
                %gr = (2)*res_vec./(s*2 + conj(res_vec).*res_vec);
                [ff, gr] = studentst(res_vec,1);
                residual = reshape(gr, anglesNumb, Detectors);
            end
            
            if (precondition == 1)
                residual = filtersinc(residual'); % filtering residual (Fourier preconditioning)
                residual = residual';
            end           
            
            x_temp = astra_create_backprojection_cuda(residual, proj_geom, vol_geom);
            X(:,:,j) = X_t(:,:,j) - (1/L_const).*x_temp;
            astra_mex_data2d('delete', sino_id);
        end
        
        if ((lambdaTV > 0) && (lambdaHO == 0))
            if (size(X,3) > 1) 
                [X] = FISTA_TV(single(X), lambdaTV, iterTV, tol); % TV regularization using FISTA
                gradTV = 1;
            else
                [X, gradTV] = FISTA_TV(single(X), lambdaTV, iterTV, tol); % TV regularization using FISTA
            end
            if (strcmp(fidelity,'student') == 1)
                objective(i) = ff + norm(gradTV(:));
            else
                objective(i) = 0.5.*norm(residual(:))^2 + norm(gradTV(:));
            end
            %  X = SplitBregman_TV(single(X), lambdaTV, iterTV, tol);  % TV-Split Bregman regularization on CPU (memory limited)
        elseif ((lambdaHO > 0) && (lambdaTV == 0))
            % Higher Order regularization
            X = LLT_model(single(X), lambdaHO, tauHO, iterHO, tol, 0);   % LLT higher order model
        elseif ((lambdaTV > 0) && (lambdaHO > 0))
            X1 = SplitBregman_TV(single(X), lambdaTV, iterTV, tol);     % TV-Split Bregman regularization on CPU (memory limited)
            X2 = LLT_model(single(X), lambdaHO, tauHO, iterHO, tol, 0);   % LLT higher order model
            X = 0.5.*(X1 + X2); % averaged combination of two solutions
        elseif ((lambdaTV == 0) && (lambdaHO == 0))
            objective(i) = 0.5.*norm(residual(:))^2;
        end
        
        
        t = (1 + sqrt(1 + 4*t^2))/2; % updating t
        X_t = X + ((t_old-1)/t).*(X - X_old); % updating X
        
        if (show == 1)
            figure(11); imshow(X(:,:,slice), [0 maxvalplot]);
            pause(0.03);            
        end
        if (strcmp(X_ideal, 'none' ) == 0)
            error(i) = RMSE(X(ROI), X_ideal(ROI));
            fprintf('%s %i %s %s %.4f  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Error RMSE:', error(i), '|', 'Objective:', objective(i));
        else
            fprintf('%s %i  %s %s %.4f \n', 'Iteration Number:', i, '|', 'Objective:', objective(i));
        end
        
        
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

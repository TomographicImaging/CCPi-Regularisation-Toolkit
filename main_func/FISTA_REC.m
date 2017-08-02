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
% 2. output - a structure with
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
    if (strcmp(proj_geom.type,'parallel') || strcmp(proj_geom.type,'parallel3d'))
        % for parallel geometry we can do just one slice
        fprintf('%s \n', 'Calculating Lipshitz constant for parallel beam geometry...');
        niter = 15; % number of iteration for the PM
        x1 = rand(N,N,1);
        sqweight = sqrt(weights(:,:,1));
        proj_geomT = proj_geom;
        proj_geomT.DetectorRowCount = 1;
        vol_geomT = vol_geom;
        vol_geomT.GridSliceCount = 1;
        [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
        y = sqweight.*y;
        astra_mex_data3d('delete', sino_id);
        
        for i = 1:niter
            [id,x1] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geomT, vol_geomT);
            s = norm(x1(:));
            x1 = x1/s;
            [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
            y = sqweight.*y;
            astra_mex_data3d('delete', sino_id);
            astra_mex_data3d('delete', id);
        end
        clear proj_geomT vol_geomT
    else
        % divergen beam geometry
        fprintf('%s \n', 'Calculating Lipshitz constant for divergen beam geometry... will take some time!');
        niter = 8; % number of iteration for PM
        x1 = rand(N,N,SlicesZ);
        sqweight = sqrt(weights);
        [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geom, vol_geom);
        y = sqweight.*y;
        astra_mex_data3d('delete', sino_id);
        
        for i = 1:niter
            [id,x1] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geom, vol_geom);
            s = norm(x1(:));
            x1 = x1/s;
            [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geom, vol_geom);
            y = sqweight.*y;
            astra_mex_data3d('delete', sino_id);
            astra_mex_data3d('delete', id);
        end
        clear x1
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
if (isfield(params,'OS'))
    % Ordered Subsets reorganisation of data and angles
    OS = 1;
    subsets = 8;
    angles = proj_geom.ProjectionAngles;
    binEdges = linspace(min(angles),max(angles),subsets+1);
    
    % assign values to bins
    [binsDiscr,~] = histc(angles, [binEdges(1:end-1) Inf]);
    
    % rearrange angles into subsets
    AnglesReorg = zeros(length(angles),1);
    SinoReorg = zeros(Detectors, anglesNumb, SlicesZ, 'single');
    
    counterM = 0;
    for ii = 1:max(binsDiscr(:))
        counter = 0;
        for jj = 1:subsets
            curr_index = ii+jj-1 + counter;
            if (binsDiscr(jj) >= ii)
                counterM = counterM + 1;
                AnglesReorg(counterM) = angles(curr_index);
                SinoReorg(:,counterM,:) = squeeze(sino(:,curr_index,:));
            end
            counter = (counter + binsDiscr(jj)) - 1;
        end
    end
    sino = SinoReorg;
    clear SinoReorg;
else 
    OS = 0; % normal FISTA
end


%----------------Reconstruction part------------------------
Resid_error = zeros(iterFISTA,1); % errors vector (if the ground truth is given)
objective = zeros(iterFISTA,1); % objective function values vector

t = 1;
X_t = X;

r = zeros(Detectors,SlicesZ, 'single'); % 2D array (for 3D data) of sparse "ring" vectors
r_x = r; % another ring variable
residual = zeros(size(sino),'single');

% Outer FISTA iterations loop
for i = 1:iterFISTA
    
    X_old = X;
    t_old = t;
    r_old = r;
    
    [sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geom, vol_geom);
    
    if (lambdaR_L1 > 0)
        % ring removal part (Group-Huber fidelity)
        for kkk = 1:anglesNumb
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
    
    objective(i) = (0.5*norm(residual(:))^2)/(Detectors*anglesNumb*SlicesZ); % for the objective function output
    
    [id, x_temp] = astra_create_backprojection3d_cuda(residual, proj_geom, vol_geom);
    
    X = X_t - (1/L_const).*x_temp;
    astra_mex_data3d('delete', sino_id);
    astra_mex_data3d('delete', id);
    
    if (lambdaFGP_TV > 0)
        % FGP-TV regularization
        [X, f_val] = FGP_TV(single(X), lambdaFGP_TV, IterationsRegul, tol, 'iso');
        objective(i) = objective(i) + f_val;
    end
    if (lambdaSB_TV > 0)
        % Split Bregman regularization
        X = SplitBregman_TV(single(X), lambdaSB_TV, IterationsRegul, tol);  % (more memory efficent)
    end
    if (lambdaHO > 0)
        % Higher Order (LLT) regularization
        X2 = LLT_model(single(X), lambdaHO, tauHO, iterHO, 3.0e-05, 0);
        X = 0.5.*(X + X2); % averaged combination of two solutions
    end
    
    
    
    if (lambdaR_L1 > 0)
        r =  max(abs(r)-lambdaR_L1, 0).*sign(r); % soft-thresholding operator for ring vector
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
        fprintf('%s %i %s %s %.4f  %s %s %f \n', 'Iteration Number:', i, '|', 'Error RMSE:', Resid_error(i), '|', 'Objective:', objective(i));
    else
        fprintf('%s %i  %s %s %f \n', 'Iteration Number:', i, '|', 'Objective:', objective(i));
    end
end
output.Resid_error = Resid_error;
output.objective = objective;
output.L_const = L_const;
output.sino = sino;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

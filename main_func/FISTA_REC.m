function [X,  output] = FISTA_REC(params)

% <<<< FISTA-based reconstruction routine using ASTRA-toolbox >>>>
% This code solves regularised PWLS problem using FISTA approach.
% The code contains multiple regularisation penalties as well as it can be
% accelerated by using ordered-subset version. Various projection
% geometries supported.

% DISCLAIMER
% It is recommended to use ASTRA version 1.8 or later in order to avoid
% crashing due to GPU memory overflow for big datasets

% ___Input___:
% params.[] file:
%----------------General Parameters------------------------
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
%       1 .Regul_Lambda_FGPTV (FGP-TV regularization parameter)
%       2 .Regul_Lambda_SBTV (SplitBregman-TV regularization parameter)
%       3 .Regul_LambdaLLT (Higher order LLT regularization parameter)
%          3.1 .Regul_tauLLT (time step parameter for LLT (HO) term)
%       4 .Regul_LambdaPatchBased_CPU (Patch-based nonlocal regularization parameter)
%          4.1  .Regul_PB_SearchW (ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window))
%          4.2  .Regul_PB_SimilW (ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window))
%          4.3  .Regul_PB_h (PB penalty function threshold)
%       5 .Regul_LambdaPatchBased_GPU (Patch-based nonlocal regularization parameter)
%          5.1  .Regul_PB_SearchW (ratio of the searching window (e.g. 3 = (2*3+1) = 7 pixels window))
%          5.2  .Regul_PB_SimilW (ratio of the similarity window (e.g. 1 = (2*1+1) = 3 pixels window))
%          5.3  .Regul_PB_h (PB penalty function threshold)
%       6 .Regul_LambdaDiffHO (Higher-Order Diffusion regularization parameter)
%          6.1  .Regul_DiffHO_EdgePar (edge-preserving noise related parameter)
%       7 .Regul_LambdaTGV (Total Generalized variation regularization parameter)
%       - .Regul_tol (tolerance to terminate regul iterations, default 1.0e-04)
%       - .Regul_Iterations (iterations for the selected penalty, default 25)
%       - .Regul_Dimension ('2D' or '3D' way to apply regularization, '3D' is the default)
%----------------Ring removal------------------------
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
            x1 = x1./s;
            [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
            y = sqweight.*y;
            astra_mex_data3d('delete', sino_id);
            astra_mex_data3d('delete', id);
        end
        %clear proj_geomT vol_geomT
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
if (isfield(params,'Regul_LambdaLLT'))
    lambdaHO = params.Regul_LambdaLLT;
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
if (isfield(params,'Regul_LambdaPatchBased_CPU'))
    lambdaPB = params.Regul_LambdaPatchBased_CPU;
else
    lambdaPB = 0;
end
if (isfield(params,'Regul_LambdaPatchBased_GPU'))
    lambdaPB_GPU = params.Regul_LambdaPatchBased_GPU;
else
    lambdaPB_GPU = 0;
end
if (isfield(params,'Regul_PB_SearchW'))
    SearchW = params.Regul_PB_SearchW;
else
    SearchW = 3; % default
end
if (isfield(params,'Regul_PB_SimilW'))
    SimilW = params.Regul_PB_SimilW;
else
    SimilW = 1; % default
end
if (isfield(params,'Regul_PB_h'))
    h_PB = params.Regul_PB_h;
else
    h_PB = 0.1; % default
end
if (isfield(params,'Regul_LambdaDiffHO'))
    LambdaDiff_HO = params.Regul_LambdaDiffHO;
else
    LambdaDiff_HO = 0;
end
if (isfield(params,'Regul_DiffHO_EdgePar'))
    LambdaDiff_HO_EdgePar = params.Regul_DiffHO_EdgePar;
else
    LambdaDiff_HO_EdgePar = 0.01;
end
if (isfield(params,'Regul_LambdaTGV'))
    LambdaTGV = params.Regul_LambdaTGV;
else
    LambdaTGV = 0;
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
if (isfield(params,'Regul_Dimension'))
    Dimension = params.Regul_Dimension;
    if ((strcmp('2D', Dimension) ~= 1) && (strcmp('3D', Dimension) ~= 1))
        Dimension = '3D';
    end
else
    Dimension = '3D';
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
if (isfield(params,'subsets'))
    % Ordered Subsets reorganisation of data and angles
    subsets = params.subsets; % subsets number
    angles = proj_geom.ProjectionAngles;
    binEdges = linspace(min(angles),max(angles),subsets+1);
    
    % assign values to bins
    [binsDiscr,~] = histc(angles, [binEdges(1:end-1) Inf]);
    
    % get rearranged subset indices
    IndicesReorg = zeros(length(angles),1);
    counterM = 0;
    for ii = 1:max(binsDiscr(:))
        counter = 0;
        for jj = 1:subsets
            curr_index = ii+jj-1 + counter;
            if (binsDiscr(jj) >= ii)
                counterM = counterM + 1;
                IndicesReorg(counterM) = curr_index;
            end
            counter = (counter + binsDiscr(jj)) - 1;
        end
    end
else
    subsets = 0; % Classical FISTA
end

%----------------Reconstruction part------------------------
Resid_error = zeros(iterFISTA,1); % errors vector (if the ground truth is given)
objective = zeros(iterFISTA,1); % objective function values vector


if (subsets == 0)
    % Classical FISTA
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
        
        % if the geometry is parallel use slice-by-slice projection-backprojection routine
        if (strcmp(proj_geom.type,'parallel') || strcmp(proj_geom.type,'parallel3d'))
            sino_updt = zeros(size(sino),'single');
            for kkk = 1:SlicesZ
                [sino_id, sino_updt(:,:,kkk)] = astra_create_sino3d_cuda(X_t(:,:,kkk), proj_geomT, vol_geomT);
                astra_mex_data3d('delete', sino_id);
            end
        else
            % for divergent 3D geometry (watch the GPU memory overflow in earlier ASTRA versions < 1.8)
            [sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geom, vol_geom);
        end
        
        if (lambdaR_L1 > 0)
            % the ring removal part (Group-Huber fidelity)
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
        
        objective(i) = (0.5*sum(residual(:).^2)); % for the objective function output
        
        % if the geometry is parallel use slice-by-slice projection-backprojection routine
        if (strcmp(proj_geom.type,'parallel') || strcmp(proj_geom.type,'parallel3d'))
            x_temp = zeros(size(X),'single');
            for kkk = 1:SlicesZ
                [id, x_temp(:,:,kkk)] = astra_create_backprojection3d_cuda(squeeze(residual(:,:,kkk)), proj_geomT, vol_geomT);
                astra_mex_data3d('delete', id);
            end
        else
            [id, x_temp] = astra_create_backprojection3d_cuda(residual, proj_geom, vol_geom);
        end
        X = X_t - (1/L_const).*x_temp;
        astra_mex_data3d('delete', sino_id);
        astra_mex_data3d('delete', id);
        
        % regularization
        if (lambdaFGP_TV > 0)
            % FGP-TV regularization
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    [X(:,:,kkk), f_val] = FGP_TV(single(X(:,:,kkk)), lambdaFGP_TV, IterationsRegul, tol, 'iso');
                end
            else
                % 3D regularization
                [X, f_val] = FGP_TV(single(X), lambdaFGP_TV, IterationsRegul, tol, 'iso');
            end
            objective(i) = (objective(i) + f_val)./(Detectors*anglesNumb*SlicesZ);
        end
        if (lambdaSB_TV > 0)
            % Split Bregman regularization
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    X(:,:,kkk) = SplitBregman_TV(single(X(:,:,kkk)), lambdaSB_TV, IterationsRegul, tol);  % (more memory efficent)
                end
            else
                % 3D regularization
                X = SplitBregman_TV(single(X), lambdaSB_TV, IterationsRegul, tol);  % (more memory efficent)
            end
        end
        if (lambdaHO > 0)
            % Higher Order (LLT) regularization
            X2 = zeros(N,N,SlicesZ,'single');
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    X2(:,:,kkk) = LLT_model(single(X(:,:,kkk)), lambdaHO, tauHO, iterHO, 3.0e-05, 0);
                end
            else
                % 3D regularization
                X2 = LLT_model(single(X), lambdaHO, tauHO, iterHO, 3.0e-05, 0);
            end
            X = 0.5.*(X + X2); % averaged combination of two solutions
            
        end
        if (lambdaPB > 0)
            % Patch-Based regularization (can be very slow on CPU)
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    X(:,:,kkk) = PatchBased_Regul(single(X(:,:,kkk)), SearchW, SimilW, h_PB, lambdaPB);
                end
            else
                X = PatchBased_Regul(single(X), SearchW, SimilW, h_PB, lambdaPB);
            end
        end
        if (lambdaPB_GPU > 0)
            % Patch-Based regularization (GPU CUDA implementation)
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    X(:,:,kkk) = NLM_GPU(single(X(:,:,kkk)), SearchW, SimilW, h_PB, lambdaPB_GPU);
                end
            else
                X = NLM_GPU(single(X), SearchW, SimilW, h_PB, lambdaPB_GPU);
            end
        end
        if (LambdaDiff_HO > 0)
            % Higher-order diffusion penalty (GPU CUDA implementation)
            if ((strcmp('2D', Dimension) == 1))
                % 2D regularization
                for kkk = 1:SlicesZ
                    X(:,:,kkk) = Diff4thHajiaboli_GPU(single(X(:,:,kkk)), LambdaDiff_HO_EdgePar, LambdaDiff_HO, IterationsRegul);
                end
            else
                X = Diff4thHajiaboli_GPU(X, LambdaDiff_HO_EdgePar, LambdaDiff_HO, IterationsRegul);
            end
        end
        if (LambdaTGV > 0)
            % Total Generalized variation (currently only 2D)
            lamTGV1 = 1.1; % smoothing trade-off parameters, see Pock's paper
            lamTGV2 = 0.8; % second-order term
            for kkk = 1:SlicesZ
                X(:,:,kkk) = TGV_PD(single(X(:,:,kkk)), LambdaTGV, lamTGV1, lamTGV2, IterationsRegul);
            end
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
else
    % Ordered Subsets (OS) FISTA reconstruction routine (normally one order of magnitude faster than classical)
    t = 1;
    X_t = X;
    proj_geomSUB = proj_geom;
    
    
    r = zeros(Detectors,SlicesZ, 'single'); % 2D array (for 3D data) of sparse "ring" vectors
    r_x = r; % another ring variable
    residual2 = zeros(size(sino),'single');
    
    % Outer FISTA iterations loop
    for i = 1:iterFISTA
        
        % With OS approach it becomes trickier to correlate independent subsets, hence additional work is required
        % one solution is to work with a full sinogram at times
        if ((i >= 3) && (lambdaR_L1 > 0))
            [sino_id2, sino_updt2] = astra_create_sino3d_cuda(X, proj_geom, vol_geom);
            astra_mex_data3d('delete', sino_id2);
        end
        
        % subsets loop
        counterInd = 1;
        for ss = 1:subsets
            X_old = X;
            t_old = t;
            r_old = r;
            
            numProjSub = binsDiscr(ss); % the number of projections per subset
            CurrSubIndeces = IndicesReorg(counterInd:(counterInd + numProjSub - 1)); % extract indeces attached to the subset
            proj_geomSUB.ProjectionAngles = angles(CurrSubIndeces);
            
            if (lambdaR_L1 > 0)
                
                % the ring removal part (Group-Huber fidelity)
                % first 2 iterations do additional work reconstructing whole dataset to ensure
                % the stablility
                if (i < 3)
                    [sino_id2, sino_updt2] = astra_create_sino3d_cuda(X_t, proj_geom, vol_geom);
                    astra_mex_data3d('delete', sino_id2);
                else
                    [sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geomSUB, vol_geom);
                end
                
                for kkk = 1:anglesNumb
                    residual2(:,kkk,:) = squeeze(weights(:,kkk,:)).*(squeeze(sino_updt2(:,kkk,:)) - (squeeze(sino(:,kkk,:)) - alpha_ring.*r_x));
                end
                
                residual = zeros(Detectors, numProjSub, SlicesZ,'single');
                for kkk = 1:numProjSub
                    indC = CurrSubIndeces(kkk);
                    if (i < 3)
                        residual(:,kkk,:) =  squeeze(residual2(:,indC,:));
                    else
                        residual(:,kkk,:) =  squeeze(weights(:,indC,:)).*(squeeze(sino_updt(:,kkk,:)) - (squeeze(sino(:,indC,:)) - alpha_ring.*r_x));
                    end
                end
                vec = sum(residual2,2);
                if (SlicesZ > 1)
                    vec = squeeze(vec(:,1,:));
                end
                r = r_x - (1./L_const).*vec;
            else
                [sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geomSUB, vol_geom);
                % no ring removal
                residual = squeeze(weights(:,CurrSubIndeces,:)).*(sino_updt - squeeze(sino(:,CurrSubIndeces,:)));
            end
            
            [id, x_temp] = astra_create_backprojection3d_cuda(residual, proj_geomSUB, vol_geom);
            
            X = X_t - (1/L_const).*x_temp;
            astra_mex_data3d('delete', sino_id);
            astra_mex_data3d('delete', id);
            
            % regularization
            if (lambdaFGP_TV > 0)
                % FGP-TV regularization
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        [X(:,:,kkk), f_val] = FGP_TV(single(X(:,:,kkk)), lambdaFGP_TV/subsets, IterationsRegul, tol, 'iso');
                    end
                else
                    % 3D regularization
                    [X, f_val] = FGP_TV(single(X), lambdaFGP_TV/subsets, IterationsRegul, tol, 'iso');
                end
                objective(i) = objective(i) + f_val;
            end
            if (lambdaSB_TV > 0)
                % Split Bregman regularization
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        X(:,:,kkk) = SplitBregman_TV(single(X(:,:,kkk)), lambdaSB_TV/subsets, IterationsRegul, tol);  % (more memory efficent)
                    end
                else
                    % 3D regularization
                    X = SplitBregman_TV(single(X), lambdaSB_TV/subsets, IterationsRegul, tol);  % (more memory efficent)
                end
            end
            if (lambdaHO > 0)
                % Higher Order (LLT) regularization
                X2 = zeros(N,N,SlicesZ,'single');
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        X2(:,:,kkk) = LLT_model(single(X(:,:,kkk)), lambdaHO/subsets, tauHO/subsets, iterHO, 2.0e-05, 0);
                    end
                else
                    % 3D regularization
                    X2 = LLT_model(single(X), lambdaHO/subsets, tauHO/subsets, iterHO, 2.0e-05, 0);
                end
                X = 0.5.*(X + X2); % the averaged combination of two solutions
            end
            if (lambdaPB > 0)
                % Patch-Based regularization (can be slow on CPU)
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        X(:,:,kkk) = PatchBased_Regul(single(X(:,:,kkk)), SearchW, SimilW, h_PB, lambdaPB/subsets);
                    end
                else
                    X = PatchBased_Regul(single(X), SearchW, SimilW, h_PB, lambdaPB/subsets);
                end
            end
            if (lambdaPB_GPU > 0)
                % Patch-Based regularization (GPU CUDA implementation)
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        X(:,:,kkk) = NLM_GPU(single(X(:,:,kkk)), SearchW, SimilW, h_PB, lambdaPB_GPU);
                    end
                else
                    X = NLM_GPU(single(X), SearchW, SimilW, h_PB, lambdaPB_GPU);
                end
            end
            if (LambdaDiff_HO > 0)
                % Higher-order diffusion penalty (GPU CUDA implementation)
                if ((strcmp('2D', Dimension) == 1))
                    % 2D regularization
                    for kkk = 1:SlicesZ
                        X(:,:,kkk) = Diff4thHajiaboli_GPU(single(X(:,:,kkk)), LambdaDiff_HO_EdgePar, LambdaDiff_HO, round(IterationsRegul/subsets));
                    end
                else
                    X = Diff4thHajiaboli_GPU(X, LambdaDiff_HO_EdgePar, LambdaDiff_HO, round(IterationsRegul/subsets));
                end
            end
            if (LambdaTGV > 0)
                % Total Generalized variation (currently only 2D)
                lamTGV1 = 1.1; % smoothing trade-off parameters, see Pock's paper
                lamTGV2 = 0.5; % second-order term
                for kkk = 1:SlicesZ
                    X(:,:,kkk) = TGV_PD(single(X(:,:,kkk)), LambdaTGV/subsets, lamTGV1, lamTGV2, IterationsRegul);
                end
            end
            
            if (lambdaR_L1 > 0)
                r =  max(abs(r)-lambdaR_L1, 0).*sign(r); % soft-thresholding operator for ring vector
            end
            
            t = (1 + sqrt(1 + 4*t^2))/2; % updating t
            X_t = X + ((t_old-1)/t).*(X - X_old); % updating X
            
            if (lambdaR_L1 > 0)
                r_x = r + ((t_old-1)/t).*(r - r_old); % updating r
            end
            
            counterInd = counterInd + numProjSub;
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
end

output.Resid_error = Resid_error;
output.objective = objective;
output.L_const = L_const;

end

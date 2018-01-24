function sino_artifacts = sino_add_artifacts(sino,artifact_type)
% function to add various distortions to the sinogram space, current
% version includes: random rings and zingers (streaks)
% Input: 
% 1. sinogram
% 2. artifact type: 'rings' or 'zingers' (streaks)


[Detectors, anglesNumb, SlicesZ] = size(sino);
fprintf('%s %i %s %i %s %i %s \n', 'Sinogram has a dimension of', Detectors, 'detectors;', anglesNumb, 'projections;', SlicesZ, 'vertical slices.');

sino_artifacts = sino;

if (strcmp(artifact_type,'rings'))
    fprintf('%s \n', 'Adding rings...');    
    NumRings = round(Detectors/20); % Number of rings relatively to the size of Detectors
    IntenOff = linspace(0.05,0.5,NumRings); % the intensity of rings in the selected range
    
    for k = 1:SlicesZ
        % generate random indices to propagate rings
        RandInd = randperm(Detectors,Detectors);
        for jj = 1:NumRings
            ind_c = RandInd(jj);
            sino_artifacts(ind_c,1:end,k) = sino_artifacts(ind_c,1:end,k) + IntenOff(jj).*sino_artifacts(ind_c,1:end,k); % generate a constant offset            
        end
        
    end
elseif (strcmp(artifact_type,'zingers'))
    fprintf('%s \n', 'Adding zingers...');
else
    fprintf('%s \n', 'Nothing selected, the same sinogram returned...');
end
end
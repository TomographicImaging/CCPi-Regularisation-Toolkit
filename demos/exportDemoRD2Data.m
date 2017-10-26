clear all
close all
%%
% % adding paths
addpath('../data/');
addpath('../main_func/'); addpath('../main_func/regularizers_CPU/');
addpath('../supp/');

load('DendrRawData.mat') % load raw data of 3D dendritic set
angles_rad = angles*(pi/180); % conversion to radians
size_det = size(data_raw3D,1); % detectors dim
angSize = size(data_raw3D, 2); % angles dim
slices_tot = size(data_raw3D, 3); % no of slices
recon_size = 950; % reconstruction size

Sino3D = zeros(size_det, angSize, slices_tot, 'single'); % log-corrected sino
% normalizing the data
for  jj = 1:slices_tot
    sino = data_raw3D(:,:,jj);
    for ii = 1:angSize
        Sino3D(:,ii,jj) = log((flats_ar(:,jj)-darks_ar(:,jj))./(single(sino(:,ii)) - darks_ar(:,jj)));
    end
end

Sino3D = Sino3D.*1000;
Weights3D = single(data_raw3D); % weights for PW model
clear data_raw3D

hdf5write('DendrData.h5', '/Weights3D', Weights3D)
hdf5write('DendrData.h5', '/Sino3D', Sino3D, 'WriteMode', 'append')
hdf5write('DendrData.h5', '/angles_rad', angles_rad,  'WriteMode', 'append')
hdf5write('DendrData.h5', '/size_det', size_det,  'WriteMode', 'append')
hdf5write('DendrData.h5', '/angSize', angSize,  'WriteMode', 'append')
hdf5write('DendrData.h5', '/slices_tot', slices_tot,  'WriteMode', 'append')
hdf5write('DendrData.h5', '/recon_size', recon_size,  'WriteMode', 'append')
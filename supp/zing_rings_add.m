%  uncomment this part of script to generate data with different noise characterisitcs

fprintf('%s\n', 'Generating Projection Data...');
multfactor = 1000;
% Creating RHS (b) - the sinogram (using a strip projection model)
vol_geom = astra_create_vol_geom(N, N);
proj_geom = astra_create_proj_geom('parallel', 1.0, P, theta_rad);
proj_id_temp = astra_create_projector('strip', proj_geom, vol_geom);
[sinogram_id, sinogramIdeal] = astra_create_sino(phantom./multfactor, proj_id_temp);
astra_mex_data2d('delete',sinogram_id);
astra_mex_algorithm('delete',proj_id_temp);
%
% % adding Gaussian noise
% eta = 0.04;  % Relative noise level
% E = randn(size(sinogram));
% sinogram = sinogram + eta*norm(sinogram,'fro')*E/norm(E,'fro');  % adding noise to the sinogram
% sinogram(sinogram<0) = 0;
% clear E;

%%
% adding zingers
val_offset = 0;
sino_zing = sinogramIdeal;
vec1 = [60, 80, 80, 70, 70, 90, 90, 40, 130, 145, 155, 125];
vec2 = [350, 450, 190, 500, 250, 530, 330, 230, 550, 250, 450, 195];
for jj = 1:length(vec1)
    for i1 = -2:2
        for j1 = -2:2
            sino_zing(vec1(jj)+i1, vec2(jj)+j1) = val_offset;
        end
    end
end

% adding stripes into the signogram
sino_zing_rings = sino_zing;
coeff = linspace2(0.01,0.15,180);
vmax = max(sinogramIdeal(:));
sino_zing_rings(1:180,120) = sino_zing_rings(1:180,120) + vmax*0.13;
sino_zing_rings(80:180,209) = sino_zing_rings(80:180,209) + vmax*0.14;
sino_zing_rings(50:110,210) = sino_zing_rings(50:110,210) + vmax*0.12;
sino_zing_rings(1:180,211) = sino_zing_rings(1:180,211) + vmax*0.14;
sino_zing_rings(1:180,300) = sino_zing_rings(1:180,300) + vmax*coeff(:);
sino_zing_rings(1:180,301) = sino_zing_rings(1:180,301) + vmax*0.14;
sino_zing_rings(10:100,302) = sino_zing_rings(10:100,302) + vmax*0.15;
sino_zing_rings(90:180,350) = sino_zing_rings(90:180,350) + vmax*0.11;
sino_zing_rings(60:140,410) = sino_zing_rings(60:140,410) + vmax*0.12;
sino_zing_rings(1:180,411) = sino_zing_rings(1:180,411) + vmax*0.14;
sino_zing_rings(1:180,412) = sino_zing_rings(1:180,412) + vmax*coeff(:);
sino_zing_rings(1:180,413) = sino_zing_rings(1:180,413) + vmax*coeff(:);
sino_zing_rings(1:180,500) = sino_zing_rings(1:180,500) - vmax*0.12;
sino_zing_rings(1:180,501) = sino_zing_rings(1:180,501) - vmax*0.12;
sino_zing_rings(1:180,550) = sino_zing_rings(1:180,550) + vmax*0.11;
sino_zing_rings(1:180,551) = sino_zing_rings(1:180,551) + vmax*0.11;
sino_zing_rings(1:180,552) = sino_zing_rings(1:180,552) + vmax*0.11;

sino_zing_rings(sino_zing_rings < 0) = 0;
%%

% adding Poisson noise
dose = 50000;
dataExp = dose.*exp(-sino_zing_rings); % noiseless raw data
dataPnoise = astra_add_noise_to_sino(dataExp,2*dose); % pre-log noisy raw data (weights)
Dweights = dataPnoise; 
sinogram = log(dose./dataPnoise);  %log corrected data -> sinogram
sinogram = abs(sinogram);
clear dataPnoise dataExp

% normalizing
sinogram = sinogram.*multfactor;
sino_zing_rings = sinogram;
Dweights = multfactor./Dweights;

%
% figure(1);
% set(gcf, 'Position', get(0,'Screensize'));
% subplot(1,2,1); imshow(phantom,[0 0.6]); title('Ideal Phantom'); colorbar;
% subplot(1,2,2); imshow(sinogram,[0 180]);  title('Noisy Sinogram');  colorbar;
% colormap(cmapnew);

% figure;
% set(gcf, 'Position', get(0,'Screensize'));
% subplot(1,2,1); imshow(sinogramIdeal,[0 180]);  title('Ideal Sinogram');  colorbar;
% imshow(sino_zing_rings,[0 180]); title('Noisy Sinogram with zingers and stripes');  colorbar;
% colormap(cmapnew);
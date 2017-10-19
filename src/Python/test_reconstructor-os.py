# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:34:49 2017

@author: ofn77899
Based on DemoRD2.m
"""

import h5py
import numpy

from ccpi.fista.FISTAReconstructor import FISTAReconstructor
import astra
import matplotlib.pyplot as plt

def RMSE(signal1, signal2):
    '''RMSE Root Mean Squared Error'''
    if numpy.shape(signal1) == numpy.shape(signal2):
        err = (signal1 - signal2)
        err = numpy.sum( err * err )/numpy.size(signal1);  # MSE
        err = sqrt(err);                                   # RMSE
        return err
    else:
        raise Exception('Input signals must have the same shape')
  
filename = r'/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/demos/DendrData.h5'
nx = h5py.File(filename, "r")
#getEntry(nx, '/')
# I have exported the entries as children of /
entries = [entry for entry in nx['/'].keys()]
print (entries)

Sino3D = numpy.asarray(nx.get('/Sino3D'), dtype="float32")
Weights3D = numpy.asarray(nx.get('/Weights3D'), dtype="float32")
angSize = numpy.asarray(nx.get('/angSize'), dtype=int)[0]
angles_rad = numpy.asarray(nx.get('/angles_rad'), dtype="float32")
recon_size = numpy.asarray(nx.get('/recon_size'), dtype=int)[0]
size_det = numpy.asarray(nx.get('/size_det'), dtype=int)[0]
slices_tot = numpy.asarray(nx.get('/slices_tot'), dtype=int)[0]

Z_slices = 20
det_row_count = Z_slices
# next definition is just for consistency of naming
det_col_count = size_det

detectorSpacingX = 1.0
detectorSpacingY = detectorSpacingX


proj_geom = astra.creators.create_proj_geom('parallel3d',
                                            detectorSpacingX,
                                            detectorSpacingY,
                                            det_row_count,
                                            det_col_count,
                                            angles_rad)

#vol_geom = astra_create_vol_geom(recon_size,recon_size,Z_slices);
image_size_x = recon_size
image_size_y = recon_size
image_size_z = Z_slices
vol_geom = astra.creators.create_vol_geom( image_size_x,
                                           image_size_y,
                                           image_size_z)

## First pass the arguments to the FISTAReconstructor and test the
## Lipschitz constant

fistaRecon = FISTAReconstructor(proj_geom,
                                vol_geom,
                                Sino3D ,
                                weights=Weights3D)

print ("Lipschitz Constant {0}".format(fistaRecon.pars['Lipschitz_constant']))
fistaRecon.setParameter(number_of_iterations = 12)
fistaRecon.setParameter(Lipschitz_constant = 767893952.0)
fistaRecon.setParameter(ring_alpha = 21)
fistaRecon.setParameter(ring_lambda_R_L1 = 0.002)

## Ordered subset
if True:
    subsets = 16
    angles = fistaRecon.getParameter('projector_geometry')['ProjectionAngles']
    #binEdges = numpy.linspace(angles.min(),
    #                          angles.max(),
    #                          subsets + 1)
    binsDiscr, binEdges = numpy.histogram(angles, bins=subsets)
    # get rearranged subset indices
    IndicesReorg = numpy.zeros((numpy.shape(angles)))
    counterM = 0
    for ii in range(binsDiscr.max()):
        counter = 0
        for jj in range(subsets):
            curr_index = ii + jj  + counter
            #print ("{0} {1} {2}".format(binsDiscr[jj] , ii, counterM))
            if binsDiscr[jj] > ii:
                if (counterM < numpy.size(IndicesReorg)):
                    IndicesReorg[counterM] = curr_index
                counterM = counterM + 1
                
            counter = counter + binsDiscr[jj] - 1


if True:
    print ("Lipschitz Constant {0}".format(fistaRecon.pars['Lipschitz_constant']))
    print ("prepare for iteration")
    fistaRecon.prepareForIteration()
    
    

    print("initializing  ...")
    if False:
        # if X doesn't exist
        #N = params.vol_geom.GridColCount
        N = vol_geom['GridColCount']
        print ("N " + str(N))
        X = numpy.zeros((N,N,SlicesZ), dtype=numpy.float)
    else:
        #X = fistaRecon.initialize()
        X = numpy.load("X.npy")
    
    print (numpy.shape(X))
    X_t = X.copy()
    print ("initialized")
    proj_geom , vol_geom, sino , \
                      SlicesZ = fistaRecon.getParameter(['projector_geometry' ,
                                                            'output_geometry',
                                                            'input_sinogram',
                                                         'SlicesZ'])

    #fistaRecon.setParameter(number_of_iterations = 3)
    iterFISTA = fistaRecon.getParameter('number_of_iterations')
    # errors vector (if the ground truth is given)
    Resid_error = numpy.zeros((iterFISTA));
    # objective function values vector
    objective = numpy.zeros((iterFISTA)); 

      
    t = 1

    ## additional for 
    proj_geomSUB = proj_geom.copy()
    fistaRecon.residual2 = numpy.zeros(numpy.shape(fistaRecon.pars['input_sinogram']))
    residual2 = fistaRecon.residual2
    sino_updt_FULL = residual.copy()
    
    print ("starting iterations")
##    % Outer FISTA iterations loop
    for i in range(fistaRecon.getParameter('number_of_iterations')):
##        % With OS approach it becomes trickier to correlate independent subsets, hence additional work is required
##        % one solution is to work with a full sinogram at times
##        if ((i >= 3) && (lambdaR_L1 > 0))
##            [sino_id2, sino_updt2] = astra_create_sino3d_cuda(X, proj_geom, vol_geom);
##            astra_mex_data3d('delete', sino_id2);
##        end
        # With OS approach it becomes trickier to correlate independent subsets,
        # hence additional work is required one solution is to work with a full
        # sinogram at times


        SlicesZ, anglesNumb, Detectors = \
                    numpy.shape(fistaRecon.getParameter('input_sinogram'))        ## https://github.com/vais-ral/CCPi-FISTA_Reconstruction/issues/4
        if (i > 1 and lambdaR_L1 > 0) :
            for kkk in range(anglesNumb):
                 
                 residual2[:,kkk,:] = (weights[:,kkk,:]).squeeze() * \
                                       ((sino_updt_FULL[:,kkk,:]).squeeze() - \
                                        (sino[:,kkk,:]).squeeze() -\
                                        (alpha_ring * r_x)
                                        )
            r_old = fistaRecon.r.copy()
            vec = residual.sum(axis = 1)
            #if SlicesZ > 1:
            #    vec = vec[:,1,:] # 1 or 0?
            r_x = fistaRecon.r_x
            fistaRecon.r = (r_x - (1./L_const) * vec).copy()

        # subset loop
        counterInd = 1
        for ss in range(fistaRecon.getParameter('subsets')):
            print ("Subset {0}".format(ss))
            X_old = X.copy()
            t_old = t
            
            # the number of projections per subset
            numProjSub = fistaRecon.getParameter('os_bins')[ss]
            CurrSubIndices = fistaRecon.getParameter('os_indices')\
                             [counterInd:counterInd+numProjSub-1]
            proj_geomSUB['ProjectionAngles'] = angles[CurrSubIndeces]
            
##        if fistaRecon.getParameter('projector_geometry')['type'] == 'parallel' or \
##           fistaRecon.getParameter('projector_geometry')['type'] == 'fanflat' or \
##           fistaRecon.getParameter('projector_geometry')['type'] == 'fanflat_vec' :
##            # if the geometry is parallel use slice-by-slice
##            # projection-backprojection routine
##            #sino_updt = zeros(size(sino),'single');
##            proj_geomT = proj_geom.copy()
##            proj_geomT['DetectorRowCount'] = 1
##            vol_geomT = vol_geom.copy()
##            vol_geomT['GridSliceCount'] = 1;
##            sino_updt = numpy.zeros(numpy.shape(sino), dtype=numpy.float)
##            for kkk in range(SlicesZ):
##                sino_id, sino_updt[kkk] = \
##                         astra.creators.create_sino3d_gpu(
##                             X_t[kkk:kkk+1], proj_geom, vol_geom)
##                astra.matlab.data3d('delete', sino_id)
##        else:
##            # for divergent 3D geometry (watch the GPU memory overflow in
##            # ASTRA versions < 1.8)
##            #[sino_id, sino_updt] = astra_create_sino3d_cuda(X_t, proj_geom, vol_geom);
##            sino_id, sino_updt = astra.creators.create_sino3d_gpu(
##                X_t, proj_geom, vol_geom)

        ## RING REMOVAL
        residual = fistaRecon.residual
        
        lambdaR_L1 , alpha_ring , weights , L_const= \
                   fistaRecon.getParameter(['ring_lambda_R_L1',
                                           'ring_alpha' , 'weights',
                                           'Lipschitz_constant'])
        sino_updt_Sub = numpy.zeros(numpy.shape(self.pars['input_sinogram']))
        if lambdaR_L1 > 0 :
             print ("ring removal")
             geometry_type = fistaRecon.getParameter('projector_geometry')['type']
             if geometry_type == 'parallel' or \
                geometry_type == 'fanflat' or \
                geometry_type == 'fanflat_vec' :
                 # here
                 for kkk in range(SlicesZ):
                     sino_id, sinoT[kkk] = \
                             astra.creators.create_sino3d_gpu(
                                 X_t[kkk:kkk+1], proj_geomSUB, vol_geom)
                     sino_updt_Sub[kkk] = sinoT.T.copy()
                     astra.matlab.data3d('delete', sino_id)

##                 % if geometry is 2D use slice-by-slice projection-backprojection routine                    
##                    for kkk = 1:SlicesZ
##                        [sino_id, sinoT] = astra_create_sino_cuda(X_t(:,:,kkk), proj_geomSUB, vol_geom);
##                        sino_updt_Sub(:,:,kkk) = sinoT';
##                        astra_mex_data2d('delete', sino_id);
##                    end
                

##             
## https://github.com/vais-ral/CCPi-FISTA_Reconstruction/issues/4
             if i < 3:
                 pass
             else:
                 sino_id, sino_updt = astra.creators.create_sino3d_gpu(
                      X_t, proj_geomSUB, vol_geom)
##                 sino_id, sino_updt = astra.creators.create_sino3d_gpu(
##                    X, proj_geom, vol_geom)
##                 astra.matlab.data3d('delete', sino_id)
                
             for kkk in range(anglesNumb):
                 
                 residual2[:,kkk,:] = (weights[:,kkk,:]).squeeze() * \
                                       ((sino_updt2[:,kkk,:]).squeeze() - \
                                        (sino[:,kkk,:]).squeeze() -\
                                        (alpha_ring * r_x)
                                        )
             shape = list(numpy.shape(fistaRecon.getParameter('input_sinogram')))
             shape[1] = numProjSub
             fistaRecon.residual = numpy.zeros(shape)
             if fistaRecon.residual.__hash__() != residual.__hash__():
                 residual = fistaRecon.residual
##             for kkk = 1:numProjSub
##                    indC = CurrSubIndeces(kkk);
##                    if (i < 3)
##                        residual(:,kkk,:) =  squeeze(residual2(:,indC,:));
##                    else
##                        residual(:,kkk,:) =  squeeze(weights(:,indC,:)).*(squeeze(sino_updt(:,kkk,:)) - (squeeze(sino(:,indC,:)) - alpha_ring.*r_x));
##                    end
##                end    
             for kk in range(numProjSub):
                 indC = fistaRecon.getParameter('os_indices')[kkk]
                 if i < 3:
                     residual[:,kkk,:] = residual2[:,indC,:].squeeze()
                 else:
                     residual[:,kkk,:] = \
                        weights[:,indC,:].squeeze() * sino_updt[:,kkk,:].squeeze() - \
                        sino[:,indC,:].squeeze() - alpha_ring * fistaRecon.r_x
                        #squeeze(weights(:,indC,:)).* \
                        #            (squeeze(sino_updt(:,kkk,:)) - \
                        #(squeeze(sino(:,indC,:)) - alpha_ring.*r_x));
                     

                
             vec = residual.sum(axis = 1)
             #if SlicesZ > 1:
             #    vec = vec[:,1,:].squeeze()
             fistaRecon.r = (r_x - (1./L_const) * vec).copy()
             objective[i] = (0.5 * (residual ** 2).sum())
##            % the ring removal part (Group-Huber fidelity)
##            for kkk = 1:anglesNumb
##                residual(:,kkk,:) =  squeeze(weights(:,kkk,:)).*
##                 (squeeze(sino_updt(:,kkk,:)) -
##                 (squeeze(sino(:,kkk,:)) - alpha_ring.*r_x));
##            end
##            vec = sum(residual,2);
##            if (SlicesZ > 1)
##                vec = squeeze(vec(:,1,:));
##            end
##            r = r_x - (1./L_const).*vec;
##            objective(i) = (0.5*sum(residual(:).^2)); % for the objective function output

        

        # Projection/Backprojection Routine
        if fistaRecon.getParameter('projector_geometry')['type'] == 'parallel' or \
           fistaRecon.getParameter('projector_geometry')['type'] == 'fanflat' or\
           fistaRecon.getParameter('projector_geometry')['type'] == 'fanflat_vec':
            x_temp = numpy.zeros(numpy.shape(X),dtype=numpy.float32)
            print ("Projection/Backprojection Routine")
            for kkk in range(SlicesZ):
                
                x_id, x_temp[kkk] = \
                         astra.creators.create_backprojection3d_gpu(
                             residual[kkk:kkk+1],
                             proj_geomT, vol_geomT)
                astra.matlab.data3d('delete', x_id)
        else:
            x_id, x_temp = \
                  astra.creators.create_backprojection3d_gpu(
                      residual, proj_geom, vol_geom)            

        X = X_t - (1/L_const) * x_temp
        astra.matlab.data3d('delete', sino_id)
        astra.matlab.data3d('delete', x_id)
        

        ## REGULARIZATION
        ## SKIPPING FOR NOW
        ## Should be simpli
        # regularizer = fistaRecon.getParameter('regularizer')
        # for slices:
        # out = regularizer(input=X)
        print ("skipping regularizer")


        ## FINAL
        print ("final")
        lambdaR_L1 = fistaRecon.getParameter('ring_lambda_R_L1')
        if lambdaR_L1 > 0:
            fistaRecon.r = numpy.max(
                numpy.abs(fistaRecon.r) - lambdaR_L1 , 0) * \
                numpy.sign(fistaRecon.r)
        t = (1 + numpy.sqrt(1 + 4 * t**2))/2
        X_t = X + (((t_old -1)/t) * (X - X_old))

        if lambdaR_L1 > 0:
            fistaRecon.r_x = fistaRecon.r + \
                             (((t_old-1)/t) * (fistaRecon.r - r_old))

        if fistaRecon.getParameter('region_of_interest') is None:
            string = 'Iteration Number {0} | Objective {1} \n'
            print (string.format( i, objective[i]))
        else:
            ROI , X_ideal = fistaRecon.getParameter('region_of_interest',
                                                    'ideal_image')
            
            Resid_error[i] = RMSE(X*ROI, X_ideal*ROI)
            string = 'Iteration Number {0} | RMS Error {1} | Objective {2} \n'
            print (string.format(i,Resid_error[i], objective[i]))
            
##        if (lambdaR_L1 > 0)
##            r =  max(abs(r)-lambdaR_L1, 0).*sign(r); % soft-thresholding operator for ring vector
##        end
##        
##        t = (1 + sqrt(1 + 4*t^2))/2; % updating t
##        X_t = X + ((t_old-1)/t).*(X - X_old); % updating X
##        
##        if (lambdaR_L1 > 0)
##            r_x = r + ((t_old-1)/t).*(r - r_old); % updating r
##        end
##        
##        if (show == 1)
##            figure(10); imshow(X(:,:,slice), [0 maxvalplot]);
##            if (lambdaR_L1 > 0)
##                figure(11); plot(r); title('Rings offset vector')
##            end
##            pause(0.01);
##        end
##        if (strcmp(X_ideal, 'none' ) == 0)
##            Resid_error(i) = RMSE(X(ROI), X_ideal(ROI));
##            fprintf('%s %i %s %s %.4f  %s %s %f \n', 'Iteration Number:', i, '|', 'Error RMSE:', Resid_error(i), '|', 'Objective:', objective(i));
##        else
##            fprintf('%s %i  %s %s %f \n', 'Iteration Number:', i, '|', 'Objective:', objective(i));
##        end
else:
    fistaRecon = FISTAReconstructor(proj_geom,
                                vol_geom,
                                Sino3D ,
                                weights=Weights3D)

    print ("Lipschitz Constant {0}".format(fistaRecon.pars['Lipschitz_constant']))
    fistaRecon.setParameter(number_of_iterations = 12)
    fistaRecon.setParameter(Lipschitz_constant = 767893952.0)
    fistaRecon.setParameter(ring_alpha = 21)
    fistaRecon.setParameter(ring_lambda_R_L1 = 0.002)
    fistaRecon.prepareForIteration()
    X = fistaRecon.iterate(numpy.load("X.npy"))

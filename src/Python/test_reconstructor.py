# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:34:49 2017

@author: ofn77899
Based on DemoRD2.m
"""

import h5py
import numpy

from ccpi.reconstruction_dev.FISTAReconstructor import FISTAReconstructor
import astra

##def getEntry(nx, location):
##    for item in nx[location].keys():
##        print (item)
  
filename = r'/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/demos/DendrData.h5'
nx = h5py.File(filename, "r")
#getEntry(nx, '/')
# I have exported the entries as children of /
entries = [entry for entry in nx['/'].keys()]
print (entries)

Sino3D = numpy.asarray(nx.get('/Sino3D'))
Weights3D = numpy.asarray(nx.get('/Weights3D'))
angSize = numpy.asarray(nx.get('/angSize'), dtype=int)[0]
angles_rad = numpy.asarray(nx.get('/angles_rad'))
recon_size = numpy.asarray(nx.get('/recon_size'), dtype=int)[0]
size_det = numpy.asarray(nx.get('/size_det'), dtype=int)[0]
slices_tot = numpy.asarray(nx.get('/slices_tot'), dtype=int)[0]

Z_slices = 3
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

#fistaRecon = FISTAReconstructor(proj_geom, vol_geom, Sino3D )
 #N = params.vol_geom.GridColCount
 
pars = dict()
pars['projector_geometry'] = proj_geom
pars['output_geometry'] = vol_geom
pars['input_sinogram'] = Sino3D
sliceZ , nangles , detectors  = numpy.shape(Sino3D)
pars['detectors'] = detectors
pars['number_of_angles'] = nangles
pars['SlicesZ'] = sliceZ
    

pars['weights'] = numpy.ones(numpy.shape(pars['input_sinogram']))
         
N = pars['output_geometry']['GridColCount']
proj_geom = pars['projector_geometry']
vol_geom = pars['output_geometry']
weights = pars['weights']
SlicesZ = pars['SlicesZ']

if (proj_geom['type'] == 'parallel') or (proj_geom['type'] == 'parallel3d'):
    #% for parallel geometry we can do just one slice
    print('Calculating Lipshitz constant for parallel beam geometry...')
    niter = 15;# % number of iteration for the PM
    #N = params.vol_geom.GridColCount;
    #x1 = rand(N,N,1);
    x1 = numpy.random.rand(1,N,N)
    #sqweight = sqrt(weights(:,:,1));
    sqweight = numpy.sqrt(weights[0])
    proj_geomT = proj_geom.copy();
    proj_geomT['DetectorRowCount'] = 1;
    vol_geomT = vol_geom.copy();
    vol_geomT['GridSliceCount'] = 1;
    
    #[sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    #a.set_title('Lipschitz')        
    for i in range(niter):
#        [id,x1] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geomT, vol_geomT);
#            s = norm(x1(:));
#            x1 = x1/s;
#            [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
#            y = sqweight.*y;
#            astra_mex_data3d('delete', sino_id);
#            astra_mex_data3d('delete', id);
        print ("iteration {0}".format(i))
        sino_id, y = astra.creators.create_sino3d_gpu(x1,
                                                  proj_geomT,
                                                  vol_geomT)
        #a=fig.add_subplot(2,1,1)
        #imgplot = plt.imshow(y[0])
        
        y = sqweight * y # element wise multiplication
        
        #b=fig.add_subplot(2,1,2)
        #imgplot = plt.imshow(x1[0])
        #plt.show()
        
        #astra_mex_data3d('delete', sino_id);
        astra.matlab.data3d('delete', sino_id)
            
        idx,x1 = astra.creators.create_backprojection3d_gpu(sqweight*y, 
                                                            proj_geomT,
                                                            vol_geomT);
        print ("shape {1} x1 {0}".format(x1.T[:4].T, numpy.shape(x1)))
        s = numpy.linalg.norm(x1)
        ### this line?
        x1 = x1/s;
        print ("x1 {0}".format(x1.T[:4].T))
        
#        ### this line?
#        sino_id, y = astra.creators.create_sino3d_gpu(x1, 
#                                                      proj_geomT, 
#                                                      vol_geomT);
#        y = sqweight * y;
        astra.matlab.data3d('delete', sino_id);
        astra.matlab.data3d('delete', idx);
    #end
    del proj_geomT
    del vol_geomT
else:
    #% divergen beam geometry
    print('Calculating Lipshitz constant for divergen beam geometry...')
    niter = 8; #% number of iteration for PM
    x1 = numpy.random.rand(SlicesZ , N , N);
    #sqweight = sqrt(weights);
    sqweight = numpy.sqrt(weights[0])
    
    sino_id, y = astra.creators.create_sino3d_gpu(x1, proj_geom, vol_geom);
    y = sqweight*y;
    #astra_mex_data3d('delete', sino_id);
    astra.matlab.data3d('delete', sino_id);
    
    for i in range(niter):
        #[id,x1] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geom, vol_geom);
        idx,x1 = astra.creators.create_backprojection3d_gpu(sqweight*y, 
                                                            proj_geom, 
                                                            vol_geom)
        s = numpy.linalg.norm(x1)
        ### this line?
        x1 = x1/s;
        ### this line?
        #[sino_id, y] = astra_create_sino3d_gpu(x1, proj_geom, vol_geom);
        sino_id, y = astra.creators.create_sino3d_gpu(x1, 
                                                      proj_geom, 
                                                      vol_geom);
        
        y = sqweight*y;
        #astra_mex_data3d('delete', sino_id);
        #astra_mex_data3d('delete', id);
        astra.matlab.data3d('delete', sino_id);
        astra.matlab.data3d('delete', idx);
    #end
    #clear x1
    del x1

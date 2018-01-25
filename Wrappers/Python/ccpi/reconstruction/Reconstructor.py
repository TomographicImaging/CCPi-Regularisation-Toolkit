# -*- coding: utf-8 -*-
###############################################################################
#This work is part of the Core Imaging Library developed by
#Visual Analytics and Imaging System Group of the Science Technology
#Facilities Council, STFC
#
#Copyright 2017 Edoardo Pasca, Srikanth Nagella
#Copyright 2017 Daniil Kazantsev
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
###############################################################################



import numpy
import h5py
from ccpi.reconstruction.parallelbeam import alg

from Regularizer import Regularizer
from enum import Enum

import astra


class Reconstructor:
    
    class Algorithm(Enum):
        CGLS = alg.cgls
        CGLS_CONV = alg.cgls_conv
        SIRT = alg.sirt
        MLEM = alg.mlem
        CGLS_TICHONOV = alg.cgls_tikhonov
        CGLS_TVREG = alg.cgls_TVreg
        FISTA = 'fista'
        
    def __init__(self, algorithm = None, projection_data = None,
                 angles = None, center_of_rotation = None , 
                 flat_field = None, dark_field = None, 
                 iterations = None, resolution = None, isLogScale = False, threads = None, 
                 normalized_projection = None):
    
        self.pars = dict()
        self.pars['algorithm'] = algorithm
        self.pars['projection_data'] = projection_data
        self.pars['normalized_projection'] = normalized_projection
        self.pars['angles'] = angles
        self.pars['center_of_rotation'] = numpy.double(center_of_rotation)
        self.pars['flat_field'] = flat_field
        self.pars['iterations'] = iterations
        self.pars['dark_field'] = dark_field
        self.pars['resolution'] = resolution
        self.pars['isLogScale'] = isLogScale
        self.pars['threads'] = threads
        if (iterations != None):
            self.pars['iterationValues'] = numpy.zeros((iterations)) 
        
        if projection_data != None and dark_field != None and flat_field != None:
            norm = self.normalize(projection_data, dark_field, flat_field, 0.1)
            self.pars['normalized_projection'] = norm
            
    
    def setPars(self, parameters):
        keys = ['algorithm','projection_data' ,'normalized_projection', \
                'angles' , 'center_of_rotation' , 'flat_field', \
                'iterations','dark_field' , 'resolution', 'isLogScale' , \
                'threads' , 'iterationValues', 'regularize']
        
        for k in keys:
            if k not in parameters.keys():
                self.pars[k] = None
            else:
                self.pars[k] = parameters[k]
                
        
    def sanityCheck(self):
        projection_data = self.pars['projection_data']
        dark_field = self.pars['dark_field']
        flat_field = self.pars['flat_field']
        angles = self.pars['angles']
        
        if projection_data != None and dark_field != None and \
            angles != None and flat_field != None:
            data_shape =  numpy.shape(projection_data)
            angle_shape = numpy.shape(angles)
            
            if angle_shape[0] != data_shape[0]:
                #raise Exception('Projections and angles dimensions do not match: %d vs %d' % \
                #                (angle_shape[0] , data_shape[0]) )
                return (False , 'Projections and angles dimensions do not match: %d vs %d' % \
                                (angle_shape[0] , data_shape[0]) )
            
            if data_shape[1:] != numpy.shape(flat_field):
                #raise Exception('Projection and flat field dimensions do not match')
                return (False , 'Projection and flat field dimensions do not match')
            if data_shape[1:] != numpy.shape(dark_field):
                #raise Exception('Projection and dark field dimensions do not match')
                return (False , 'Projection and dark field dimensions do not match')
            
            return (True , '' )
        elif self.pars['normalized_projection'] != None:
            data_shape =  numpy.shape(self.pars['normalized_projection'])
            angle_shape = numpy.shape(angles)
            
            if angle_shape[0] != data_shape[0]:
                #raise Exception('Projections and angles dimensions do not match: %d vs %d' % \
                #                (angle_shape[0] , data_shape[0]) )
                return (False , 'Projections and angles dimensions do not match: %d vs %d' % \
                                (angle_shape[0] , data_shape[0]) )
            else:
                return (True , '' )
        else:
            return (False , 'Not enough data')
            
    def reconstruct(self, parameters = None):
        if parameters != None:
            self.setPars(parameters)
        
        go , reason = self.sanityCheck()
        if go:
            return self._reconstruct()
        else:
            raise Exception(reason)
            
            
    def _reconstruct(self, parameters=None):
        if parameters!=None:
            self.setPars(parameters)
        parameters = self.pars
        
        if parameters['algorithm'] != None and \
           parameters['normalized_projection'] != None and \
           parameters['angles'] != None and \
           parameters['center_of_rotation'] != None and \
           parameters['iterations'] != None and \
           parameters['resolution'] != None and\
           parameters['threads'] != None and\
           parameters['isLogScale'] != None:
               
               
           if parameters['algorithm'] in (Reconstructor.Algorithm.CGLS,
                        Reconstructor.Algorithm.MLEM, Reconstructor.Algorithm.SIRT):
               #store parameters
               self.pars = parameters
               result = parameters['algorithm'](
                           parameters['normalized_projection'] ,
                           parameters['angles'],
                           parameters['center_of_rotation'],
                           parameters['resolution'],
                           parameters['iterations'],
                           parameters['threads'] ,
                           parameters['isLogScale']
                           )
               return result
           elif parameters['algorithm'] in (Reconstructor.Algorithm.CGLS_CONV,
                          Reconstructor.Algorithm.CGLS_TICHONOV, 
                          Reconstructor.Algorithm.CGLS_TVREG) :
               self.pars = parameters
               result = parameters['algorithm'](
                           parameters['normalized_projection'] ,
                           parameters['angles'],
                           parameters['center_of_rotation'],
                           parameters['resolution'],
                           parameters['iterations'],
                           parameters['threads'] ,
                           parameters['regularize'],
                           numpy.zeros((parameters['iterations'])),
                           parameters['isLogScale']
                           )
               
           elif parameters['algorithm'] == Reconstructor.Algorithm.FISTA:
               pass
             
        else:
           if parameters['projection_data'] != None and \
                     parameters['dark_field'] != None and \
                     parameters['flat_field'] != None:
               norm = self.normalize(parameters['projection_data'],
                                   parameters['dark_field'], 
                                   parameters['flat_field'], 0.1)
               self.pars['normalized_projection'] = norm
               return self._reconstruct(parameters)
              
                
                
    def _normalize(self, projection, dark, flat, def_val=0):
        a = (projection - dark)
        b = (flat-dark)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            c = numpy.true_divide( a, b )
            c[ ~ numpy.isfinite( c )] = def_val  # set to not zero if 0/0 
        return c
    
    def normalize(self, projections, dark, flat, def_val=0):
        norm = [self._normalize(projection, dark, flat, def_val) for projection in projections]
        return numpy.asarray (norm, dtype=numpy.float32)
        
    
    
class FISTA():
    '''FISTA-based reconstruction algorithm using ASTRA-toolbox
    
    '''
    # <<<< FISTA-based reconstruction algorithm using ASTRA-toolbox >>>>
    # ___Input___:
    # params.[] file:
    #       - .proj_geom (geometry of the projector) [required]
    #       - .vol_geom (geometry of the reconstructed object) [required]
    #       - .sino (vectorized in 2D or 3D sinogram) [required]
    #       - .iterFISTA (iterations for the main loop, default 40)
    #       - .L_const (Lipschitz constant, default Power method)                                                                                                    )
    #       - .X_ideal (ideal image, if given)
    #       - .weights (statisitcal weights, size of the sinogram)
    #       - .ROI (Region-of-interest, only if X_ideal is given)
    #       - .initialize (a 'warm start' using SIRT method from ASTRA)
    #----------------Regularization choices------------------------
    #       - .Regul_Lambda_FGPTV (FGP-TV regularization parameter)
    #       - .Regul_Lambda_SBTV (SplitBregman-TV regularization parameter)
    #       - .Regul_Lambda_TVLLT (Higher order SB-LLT regularization parameter)
    #       - .Regul_tol (tolerance to terminate regul iterations, default 1.0e-04)
    #       - .Regul_Iterations (iterations for the selected penalty, default 25)
    #       - .Regul_tauLLT (time step parameter for LLT term)
    #       - .Ring_LambdaR_L1 (regularization parameter for L1-ring minimization, if lambdaR_L1 > 0 then switch on ring removal)
    #       - .Ring_Alpha (larger values can accelerate convergence but check stability, default 1)
    #----------------Visualization parameters------------------------
    #       - .show (visualize reconstruction 1/0, (0 default))
    #       - .maxvalplot (maximum value to use for imshow[0 maxvalplot])
    #       - .slice (for 3D volumes - slice number to imshow)
    # ___Output___:
    # 1. X - reconstructed image/volume
    # 2. output - a structure with
    #    - .Resid_error - residual error (if X_ideal is given)
    #    - .objective: value of the objective function
    #    - .L_const: Lipshitz constant to avoid recalculations
    
    # References:
    # 1. "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
    # Problems" by A. Beck and M Teboulle
    # 2. "Ring artifacts correction in compressed sensing..." by P. Paleo
    # 3. "A novel tomographic reconstruction method based on the robust
    # Student's t function for suppressing data outliers" D. Kazantsev et.al.
    # D. Kazantsev, 2016-17
    def __init__(self, projector_geometry, output_geometry, input_sinogram, **kwargs):
        self.params = dict()
        self.params['projector_geometry'] = projector_geometry
        self.params['output_geometry'] = output_geometry
        self.params['input_sinogram'] = input_sinogram
        detectors, nangles, sliceZ = numpy.shape(input_sinogram)
        self.params['detectors'] = detectors
        self.params['number_og_angles'] = nangles
        self.params['SlicesZ'] = sliceZ
        
        # Accepted input keywords
        kw = ('number_of_iterations', 'Lipschitz_constant' , 'ideal_image' ,
              'weights' , 'region_of_interest' , 'initialize' , 
              'regularizer' , 
              'ring_lambda_R_L1',
              'ring_alpha')
        
        # handle keyworded parameters
        if kwargs is not None:
            for key, value in kwargs.items():
                if key in kw:
                    #print("{0} = {1}".format(key, value))                        
                    self.pars[key] = value
                    
        # set the default values for the parameters if not set
        if 'number_of_iterations' in kwargs.keys():
            self.pars['number_of_iterations'] = kwargs['number_of_iterations']
        else:
            self.pars['number_of_iterations'] = 40
        if 'weights' in kwargs.keys():
            self.pars['weights'] = kwargs['weights']
        else:
            self.pars['weights'] = numpy.ones(numpy.shape(self.params['input_sinogram']))
        if 'Lipschitz_constant' in kwargs.keys():
            self.pars['Lipschitz_constant'] = kwargs['Lipschitz_constant']
        else:
            self.pars['Lipschitz_constant'] = self.calculateLipschitzConstantWithPowerMethod()
        
        if not self.pars['ideal_image'] in kwargs.keys():
            self.pars['ideal_image'] = None
        
        if not self.pars['region_of_interest'] :
            if self.pars['ideal_image'] == None:
                pass
            else:
                self.pars['region_of_interest'] = numpy.nonzero(self.pars['ideal_image']>0.0)
            
        if not self.pars['regularizer'] :
            self.pars['regularizer'] = None
        else:
            # the regularizer must be a correctly instantiated object
            if not self.pars['ring_lambda_R_L1']:
                self.pars['ring_lambda_R_L1'] = 0
            if not self.pars['ring_alpha']:
                self.pars['ring_alpha'] = 1
        
            
            
        
    def calculateLipschitzConstantWithPowerMethod(self):
        ''' using Power method (PM) to establish L constant'''
        
        #N = params.vol_geom.GridColCount
        N = self.pars['output_geometry'].GridColCount
        proj_geom = self.params['projector_geometry']
        vol_geom = self.params['output_geometry']
        weights = self.pars['weights']
        SlicesZ = self.pars['SlicesZ']
        
        if (proj_geom['type'] == 'parallel') or (proj_geom['type'] == 'parallel3d'):
            #% for parallel geometry we can do just one slice
            #fprintf('%s \n', 'Calculating Lipshitz constant for parallel beam geometry...');
            niter = 15;# % number of iteration for the PM
            #N = params.vol_geom.GridColCount;
            #x1 = rand(N,N,1);
            x1 = numpy.random.rand(1,N,N)
            #sqweight = sqrt(weights(:,:,1));
            sqweight = numpy.sqrt(weights.T[0])
            proj_geomT = proj_geom.copy();
            proj_geomT.DetectorRowCount = 1;
            vol_geomT = vol_geom.copy();
            vol_geomT['GridSliceCount'] = 1;
            
            
            for i in range(niter):
                if i == 0:
                    #[sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
                    sino_id, y = astra.creators.create_sino3d_gpu(x1, proj_geomT, vol_geomT);
                    y = sqweight * y # element wise multiplication
                    #astra_mex_data3d('delete', sino_id);
                    astra.matlab.data3d('delete', sino_id)
                    
                idx,x1 = astra.creators.create_backprojection3d_gpu(sqweight*y, proj_geomT, vol_geomT);
                s = numpy.linalg.norm(x1)
                ### this line?
                x1 = x1/s;
                ### this line?
                sino_id, y = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
                y = sqweight*y;
                astra.matlab.data3d('delete', sino_id);
                astra.matlab.data3d('delete', idx);
            #end
            del proj_geomT
            del vol_geomT
        else:
            #% divergen beam geometry
            #fprintf('%s \n', 'Calculating Lipshitz constant for divergen beam geometry...');
            niter = 8; #% number of iteration for PM
            x1 = numpy.random.rand(SlicesZ , N , N);
            #sqweight = sqrt(weights);
            sqweight = numpy.sqrt(weights.T[0])
            
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
        
        return s
    
    
    def setRegularizer(self, regularizer):
        #if regularizer
        self.pars['regularizer'] = regularizer
        
    
    


def getEntry(location):
    for item in nx[location].keys():
        print (item)


print ("Loading Data")

##fname = "D:\\Documents\\Dataset\\IMAT\\20170419_crabtomo\\crabtomo\\Sample\\IMAT00005153_crabstomo_Sample_000.tif"
####ind = [i * 1049 for i in range(360)]
#### use only 360 images
##images = 200
##ind = [int(i * 1049 / images) for i in range(images)]
##stack_image = dxchange.reader.read_tiff_stack(fname, ind, digit=None, slc=None)

#fname = "D:\\Documents\\Dataset\\CGLS\\24737_fd.nxs"
fname = "C:\\Users\\ofn77899\\Documents\\CCPi\\CGLS\\24737_fd_2.nxs"
nx = h5py.File(fname, "r")

# the data are stored in a particular location in the hdf5
for item in nx['entry1/tomo_entry/data'].keys():
    print (item)

data = nx.get('entry1/tomo_entry/data/rotation_angle')
angles = numpy.zeros(data.shape)
data.read_direct(angles)
print (angles)
# angles should be in degrees

data = nx.get('entry1/tomo_entry/data/data')
stack = numpy.zeros(data.shape)
data.read_direct(stack)
print (data.shape)

print ("Data Loaded")


# Normalize
data = nx.get('entry1/tomo_entry/instrument/detector/image_key')
itype = numpy.zeros(data.shape)
data.read_direct(itype)
# 2 is dark field
darks = [stack[i] for i in range(len(itype)) if itype[i] == 2 ]
dark = darks[0]
for i in range(1, len(darks)):
    dark += darks[i]
dark = dark / len(darks)
#dark[0][0] = dark[0][1]

# 1 is flat field
flats = [stack[i] for i in range(len(itype)) if itype[i] == 1 ]
flat = flats[0]
for i in range(1, len(flats)):
    flat += flats[i]
flat = flat / len(flats)
#flat[0][0] = dark[0][1]


# 0 is projection data
proj = [stack[i] for i in range(len(itype)) if itype[i] == 0 ]
angle_proj = [angles[i] for i in range(len(itype)) if itype[i] == 0 ]
angle_proj = numpy.asarray (angle_proj)
angle_proj = angle_proj.astype(numpy.float32)

# normalized data are
# norm = (projection - dark)/(flat-dark)

def normalize(projection, dark, flat, def_val=0.1):
    a = (projection - dark)
    b = (flat-dark)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        c = numpy.true_divide( a, b )
        c[ ~ numpy.isfinite( c )] = def_val  # set to not zero if 0/0 
    return c
    

norm = [normalize(projection, dark, flat) for projection in proj]
norm = numpy.asarray (norm)
norm = norm.astype(numpy.float32)

#recon = Reconstructor(algorithm = Algorithm.CGLS, normalized_projection = norm,
#                 angles = angle_proj, center_of_rotation = 86.2 , 
#                 flat_field = flat, dark_field = dark, 
#                 iterations = 15, resolution = 1, isLogScale = False, threads = 3)

#recon = Reconstructor(algorithm = Reconstructor.Algorithm.CGLS, projection_data = proj,
#                 angles = angle_proj, center_of_rotation = 86.2 , 
#                 flat_field = flat, dark_field = dark, 
#                 iterations = 15, resolution = 1, isLogScale = False, threads = 3)
#img_cgls = recon.reconstruct()
#
#pars = dict()
#pars['algorithm'] = Reconstructor.Algorithm.SIRT
#pars['projection_data'] = proj
#pars['angles'] = angle_proj
#pars['center_of_rotation'] = numpy.double(86.2)
#pars['flat_field'] = flat
#pars['iterations'] = 15
#pars['dark_field'] = dark
#pars['resolution'] = 1
#pars['isLogScale'] = False
#pars['threads'] = 3
#
#img_sirt = recon.reconstruct(pars)
#
#recon.pars['algorithm'] = Reconstructor.Algorithm.MLEM
#img_mlem = recon.reconstruct()

############################################################
############################################################
#recon.pars['algorithm'] = Reconstructor.Algorithm.CGLS_CONV
#recon.pars['regularize'] = numpy.double(0.1)
#img_cgls_conv = recon.reconstruct()

niterations = 15
threads = 3

img_cgls = alg.cgls(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads, False)
img_mlem = alg.mlem(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads, False)
img_sirt = alg.sirt(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads, False)

iteration_values = numpy.zeros((niterations,))
img_cgls_conv = alg.cgls_conv(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads,
                              iteration_values, False)
print ("iteration values %s" % str(iteration_values))

iteration_values = numpy.zeros((niterations,))
img_cgls_tikhonov = alg.cgls_tikhonov(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads,
                                      numpy.double(1e-5), iteration_values , False)
print ("iteration values %s" % str(iteration_values))
iteration_values = numpy.zeros((niterations,))
img_cgls_TVreg = alg.cgls_TVreg(norm, angle_proj, numpy.double(86.2), 1 , niterations, threads,
                                      numpy.double(1e-5), iteration_values , False)
print ("iteration values %s" % str(iteration_values))


##numpy.save("cgls_recon.npy", img_data)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,6,sharey=True)
ax[0].imshow(img_cgls[80])
ax[0].axis('off')  # clear x- and y-axes
ax[1].imshow(img_sirt[80])
ax[1].axis('off')  # clear x- and y-axes
ax[2].imshow(img_mlem[80])
ax[2].axis('off')  # clear x- and y-axesplt.show()
ax[3].imshow(img_cgls_conv[80])
ax[3].axis('off')  # clear x- and y-axesplt.show()
ax[4].imshow(img_cgls_tikhonov[80])
ax[4].axis('off')  # clear x- and y-axesplt.show()
ax[5].imshow(img_cgls_TVreg[80])
ax[5].axis('off')  # clear x- and y-axesplt.show()


plt.show()

#viewer = edo.CILViewer()
#viewer.setInputAsNumpy(img_cgls2)
#viewer.displaySliceActor(0)
#viewer.startRenderLoop()

import vtk

def NumpyToVTKImageData(numpyarray):
    if (len(numpy.shape(numpyarray)) == 3):
        doubleImg = vtk.vtkImageData()
        shape = numpy.shape(numpyarray)
        doubleImg.SetDimensions(shape[0], shape[1], shape[2])
        doubleImg.SetOrigin(0,0,0)
        doubleImg.SetSpacing(1,1,1)
        doubleImg.SetExtent(0, shape[0]-1, 0, shape[1]-1, 0, shape[2]-1)
        #self.img3D.SetScalarType(vtk.VTK_UNSIGNED_SHORT, vtk.vtkInformation())
        doubleImg.AllocateScalars(vtk.VTK_DOUBLE,1)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    doubleImg.SetScalarComponentFromDouble(
                        i,j,k,0, numpyarray[i][j][k])
    #self.setInput3DData( numpy_support.numpy_to_vtk(numpyarray) )
        # rescale to appropriate VTK_UNSIGNED_SHORT
        stats = vtk.vtkImageAccumulate()
        stats.SetInputData(doubleImg)
        stats.Update()
        iMin = stats.GetMin()[0]
        iMax = stats.GetMax()[0]
        scale = vtk.VTK_UNSIGNED_SHORT_MAX / (iMax - iMin)

        shiftScaler = vtk.vtkImageShiftScale ()
        shiftScaler.SetInputData(doubleImg)
        shiftScaler.SetScale(scale)
        shiftScaler.SetShift(iMin)
        shiftScaler.SetOutputScalarType(vtk.VTK_UNSIGNED_SHORT)
        shiftScaler.Update()
        return shiftScaler.GetOutput()
        
#writer = vtk.vtkMetaImageWriter()
#writer.SetFileName(alg + "_recon.mha")
#writer.SetInputData(NumpyToVTKImageData(img_cgls2))
#writer.Write()

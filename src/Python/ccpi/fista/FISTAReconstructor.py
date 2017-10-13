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
#from ccpi.reconstruction.parallelbeam import alg

#from ccpi.imaging.Regularizer import Regularizer
from enum import Enum

import astra

   
    
class FISTAReconstructor():
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
    def __init__(self, projector_geometry, output_geometry, input_sinogram,
                 **kwargs):
        # handle parmeters:
        # obligatory parameters
        self.pars = dict()
        self.pars['projector_geometry'] = projector_geometry
        self.pars['output_geometry'] = output_geometry
        self.pars['input_sinogram'] = input_sinogram
        detectors, nangles, sliceZ = numpy.shape(input_sinogram)
        self.pars['detectors'] = detectors
        self.pars['number_og_angles'] = nangles
        self.pars['SlicesZ'] = sliceZ

        print (self.pars)
        # handle optional input parameters (at instantiation)
        
        # Accepted input keywords
        kw = ('number_of_iterations', 
              'Lipschitz_constant' , 
              'ideal_image' ,
              'weights' , 
              'region_of_interest' , 
              'initialize' , 
              'regularizer' , 
              'ring_lambda_R_L1',
              'ring_alpha')
        self.acceptedInputKeywords = kw
        
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
            self.pars['weights'] = \
                                 numpy.ones(numpy.shape(
                                     self.pars['input_sinogram']))
        if 'Lipschitz_constant' in kwargs.keys():
            self.pars['Lipschitz_constant'] = kwargs['Lipschitz_constant']
        else:
            self.pars['Lipschitz_constant'] = \
                            self.calculateLipschitzConstantWithPowerMethod()
        
        if not 'ideal_image' in kwargs.keys():
            self.pars['ideal_image'] = None
        
        if not 'region_of_interest'in kwargs.keys() :
            if self.pars['ideal_image'] == None:
                pass
            else:
                self.pars['region_of_interest'] = numpy.nonzero(
                    self.pars['ideal_image']>0.0)
            
        if not 'regularizer' in kwargs.keys() :
            self.pars['regularizer'] = None
        else:
            # the regularizer must be a correctly instantiated object
            if not 'ring_lambda_R_L1' in kwargs.keys():
                self.pars['ring_lambda_R_L1'] = 0
            if not 'ring_alpha' in kwargs.keys():
                self.pars['ring_alpha'] = 1
        
            
            
    def setParameter(self, **kwargs):
        '''set named parameter for the regularization engine
        
        raises Exception if the named parameter is not recognized
        Typical usage is:
            
        reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
        reg.setParameter(input=u0)    
        reg.setParameter(regularization_parameter=10.)
        
        it can be also used as
        reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
        reg.setParameter(input=u0 , regularization_parameter=10.)
        '''
        
        for key , value in kwargs.items():
            if key in self.acceptedInputKeywords.keys():
                self.pars[key] = value
            else:
                raise Exception('Wrong parameter {0} for '.format(key) + 
                                'Reconstruction algorithm')
    # setParameter
    
    def calculateLipschitzConstantWithPowerMethod(self):
        ''' using Power method (PM) to establish L constant'''
        
        N = self.pars['output_geometry']['GridColCount']
        proj_geom = self.pars['projector_geometry']
        vol_geom = self.pars['output_geometry']
        weights = self.pars['weights']
        SlicesZ = self.pars['SlicesZ']
        
            
                               
        if (proj_geom['type'] == 'parallel') or \
           (proj_geom['type'] == 'parallel3d'):
            #% for parallel geometry we can do just one slice
            #print('Calculating Lipshitz constant for parallel beam geometry...')
            niter = 5;# % number of iteration for the PM
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
            
            
            for i in range(niter):
            #        [id,x1] = astra_create_backprojection3d_cuda(sqweight.*y, proj_geomT, vol_geomT);
            #            s = norm(x1(:));
            #            x1 = x1/s;
            #            [sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);
            #            y = sqweight.*y;
            #            astra_mex_data3d('delete', sino_id);
            #            astra_mex_data3d('delete', id);
                #print ("iteration {0}".format(i))
                                
                sino_id, y = astra.creators.create_sino3d_gpu(x1,
                                                          proj_geomT,
                                                          vol_geomT)
                
                y = (sqweight * y).copy() # element wise multiplication
                
                #b=fig.add_subplot(2,1,2)
                #imgplot = plt.imshow(x1[0])
                #plt.show()
                
                #astra_mex_data3d('delete', sino_id);
                astra.matlab.data3d('delete', sino_id)
                del x1
                    
                idx,x1 = astra.creators.create_backprojection3d_gpu((sqweight*y).copy(), 
                                                                    proj_geomT,
                                                                    vol_geomT)
                del y
                
                                                                    
                s = numpy.linalg.norm(x1)
                ### this line?
                x1 = (x1/s).copy();
                
            #        ### this line?
            #        sino_id, y = astra.creators.create_sino3d_gpu(x1, 
            #                                                      proj_geomT, 
            #                                                      vol_geomT);
            #        y = sqweight * y;
                astra.matlab.data3d('delete', sino_id);
                astra.matlab.data3d('delete', idx)
                print ("iteration {0} s= {1}".format(i,s))
                
            #end
            del proj_geomT
            del vol_geomT
            #plt.show()
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

        
        return s
    
    
    def setRegularizer(self, regularizer):
        if regularizer is not None:
            self.pars['regularizer'] = regularizer
        
    
    

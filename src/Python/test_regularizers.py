# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:10:05 2017

@author: ofn77899
"""

from ccpi.viewer.CILViewer2D import Converter
import vtk

import regularizers
import matplotlib.pyplot as plt
import numpy as np
import os    
from enum import Enum

class Regularizer():
    '''Class to handle regularizer algorithms to be used during reconstruction
    
    Currently 5 regularization algorithms are available:
        
    1) SplitBregman_TV
    2) FGP_TV
    3)
    4)
    5)
    
    Usage:
        the regularizer can be invoked as object or as static method
        Depending on the actual regularizer the input parameter may vary, and 
        a different default setting is defined.
        reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)

        out = reg(input=u0, regularization_parameter=10., number_of_iterations=30,
          tolerance_constant=1e-4, 
          TV_Penalty=Regularizer.TotalVariationPenalty.l1)

        out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10.,
          number_of_iterations=30, tolerance_constant=1e-4, 
          TV_Penalty=Regularizer.TotalVariationPenalty.l1)
        
        A number of optional parameters can be passed or skipped
        out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10. )

    '''
    class Algorithm(Enum):
        SplitBregman_TV = regularizers.SplitBregman_TV
        FGP_TV = regularizers.FGP_TV
        LLT_model = regularizers.LLT_model
    # Algorithm
    
    class TotalVariationPenalty(Enum):
        isotropic = 0
        l1 = 1
    # TotalVariationPenalty
        
    def __init__(self , algorithm):
        
        self.algorithm = algorithm
        self.pars = self.parsForAlgorithm(algorithm)
    # __init__
        
    def parsForAlgorithm(self, algorithm):
        pars = dict()
        if algorithm == Regularizer.Algorithm.SplitBregman_TV :
            pars['algorithm'] = algorithm
            pars['input'] = None
            pars['regularization_parameter'] = None
            pars['number_of_iterations'] = 35
            pars['tolerance_constant'] = 0.0001
            pars['TV_penalty'] = Regularizer.TotalVariationPenalty.isotropic
        elif algorithm == Regularizer.Algorithm.FGP_TV :
            pars['algorithm'] = algorithm
            pars['input'] = None
            pars['regularization_parameter'] = None
            pars['number_of_iterations'] = 50
            pars['tolerance_constant'] = 0.001
            pars['TV_penalty'] = Regularizer.TotalVariationPenalty.isotropic
        elif algorithm == Regularizer.Algorithm.LLT_model:
            pars['algorithm'] = algorithm
            pars['input'] = None
            pars['regularization_parameter'] = None
            pars['time_step'] = None
            pars['number_of_iterations'] = None
            pars['tolerance_constant'] = None
            pars['restrictive_Z_smoothing'] = 0
            
        return pars
    # parsForAlgorithm
        
    def __call__(self, input, regularization_parameter, **kwargs):
        
        if kwargs is not None:
            for key, value in kwargs.items():
                #print("{0} = {1}".format(key, value))
                self.pars[key] = value
        self.pars['input'] = input
        self.pars['regularization_parameter'] = regularization_parameter
        #for key, value in self.pars.items():
        #        print("{0} = {1}".format(key, value))
                
        if self.algorithm == Regularizer.Algorithm.SplitBregman_TV :
            return self.algorithm(input, regularization_parameter,
                              self.pars['number_of_iterations'],
                              self.pars['tolerance_constant'],
                              self.pars['TV_penalty'].value )    
        elif self.algorithm == Regularizer.Algorithm.FGP_TV :
            return self.algorithm(input, regularization_parameter,
                              self.pars['number_of_iterations'],
                              self.pars['tolerance_constant'],
                              self.pars['TV_penalty'].value )
        elif self.algorithm == Regularizer.Algorithm.LLT_model :
            #LLT_model(np::ndarray input, double d_lambda, double d_tau, int iter, double d_epsil, int switcher)
            # no default
            if None in self.pars:
                raise Exception("Not all parameters have been provided")
            else:
                return self.algorithm(input, 
                                  regularization_parameter,
                                  self.pars['time_step'] , 
                                  self.pars['number_of_iterations'],
                                  self.pars['tolerance_constant'],
                                  self.pars['restrictive_Z_smoothing'] )
            
        
    # __call__
    
    @staticmethod
    def SplitBregman_TV(input, regularization_parameter , **kwargs):
        reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
        out = list( reg(input, regularization_parameter, **kwargs) )
        out.append(reg.pars)
        return out
        
    @staticmethod
    def FGP_TV(input, regularization_parameter , **kwargs):
        reg = Regularizer(Regularizer.Algorithm.FGP_TV)
        out = list( reg(input, regularization_parameter, **kwargs) )
        out.append(reg.pars)
        return out
    
    @staticmethod
    def LLT_model(input, regularization_parameter , time_step, number_of_iterations,
                  tolerance_constant, restrictive_Z_smoothing=0):
        reg = Regularizer(Regularizer.Algorithm.FGP_TV)
        out = list( reg(input, regularization_parameter, time_step=time_step, 
                        number_of_iterations=number_of_iterations,
                        tolerance_constant=tolerance_constant, 
                        restrictive_Z_smoothing=restrictive_Z_smoothing) )
        out.append(reg.pars)
        return out
        

#Example:
# figure;
# Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
# u = SplitBregman_TV(single(u0), 10, 30, 1e-04);

filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-FISTA_reconstruction\data\lena_gray_512.tif"
reader = vtk.vtkTIFFReader()
reader.SetFileName(os.path.normpath(filename))
reader.Update()
#vtk returns 3D images, let's take just the one slice there is as 2D
Im = Converter.vtk2numpy(reader.GetOutput()).T[0]/255

#imgplot = plt.imshow(Im)
perc = 0.05
u0 = Im + (perc* np.random.normal(size=np.shape(Im)))
# map the u0 u0->u0>0
f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = f(u0).astype('float32')

# plot 
fig = plt.figure()
a=fig.add_subplot(2,3,1)
a.set_title('Original')
imgplot = plt.imshow(Im)

a=fig.add_subplot(2,3,2)
a.set_title('noise')
imgplot = plt.imshow(u0)


##############################################################################
# Call regularizer

####################### SplitBregman_TV #####################################
# u = SplitBregman_TV(single(u0), 10, 30, 1e-04);

reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)

out = reg(input=u0, regularization_parameter=10., #number_of_iterations=30,
          #tolerance_constant=1e-4, 
          TV_Penalty=Regularizer.TotalVariationPenalty.l1)

out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10., number_of_iterations=30,
          tolerance_constant=1e-4, 
          TV_Penalty=Regularizer.TotalVariationPenalty.l1)
out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10. )
pars = out2[2]

a=fig.add_subplot(2,3,3)
a.set_title('SplitBregman_TV')
textstr = 'regularization_parameter=%.2f\niterations=%d\ntolerance=%.2e\npenalty=%s'
textstr = textstr % (pars['regularization_parameter'], 
                     pars['number_of_iterations'], 
                     pars['tolerance_constant'],
                     pars['TV_penalty'].name)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(out2[0])

###################### FGP_TV #########################################
# u = FGP_TV(single(u0), 0.05, 100, 1e-04);
out2 = Regularizer.FGP_TV(input=u0, regularization_parameter=0.05,
                          number_of_iterations=10)
pars = out2[-1]

a=fig.add_subplot(2,3,4)
a.set_title('FGP_TV')
textstr = 'regularization_parameter=%.2f\niterations=%d\ntolerance=%.2e\npenalty=%s'
textstr = textstr % (pars['regularization_parameter'], 
                     pars['number_of_iterations'], 
                     pars['tolerance_constant'],
                     pars['TV_penalty'].name)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(out2[0])

###################### LLT_model #########################################
# * u0 = Im + .03*randn(size(Im)); % adding noise
# [Den] = LLT_model(single(u0), 10, 0.1, 1);
out2 = Regularizer.LLT_model(input=u0, regularization_parameter=10.,
                          time_step=0.1,
                          tolerance_constant=1e-4,
                          number_of_iterations=10)
pars = out2[-1]

a=fig.add_subplot(2,3,5)
a.set_title('LLT_model')
textstr = 'regularization_parameter=%.2f\niterations=%d\ntolerance=%.2e\ntime-step=%f'
textstr = textstr % (pars['regularization_parameter'], 
                     pars['number_of_iterations'], 
                     pars['tolerance_constant'],
                     pars['time_step']
                     )

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(out2[0])




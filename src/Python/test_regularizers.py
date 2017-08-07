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
        PatchBased_Regul = regularizers.PatchBased_Regul
        TGV_PD = regularizers.TGV_PD
    # Algorithm
    
    class TotalVariationPenalty(Enum):
        isotropic = 0
        l1 = 1
    # TotalVariationPenalty
        
    def __init__(self , algorithm):
        self.setAlgorithm ( algorithm )
    # __init__
    
    def setAlgorithm(self, algorithm):
        self.algorithm = algorithm
        self.pars = self.parsForAlgorithm(algorithm)
    # setAlgorithm
        
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
            
        elif algorithm == Regularizer.Algorithm.PatchBased_Regul:
            pars['algorithm'] = algorithm
            pars['input'] = None
            pars['searching_window_ratio'] = None
            pars['similarity_window_ratio'] = None
            pars['PB_filtering_parameter'] = None
            pars['regularization_parameter'] = None
            
        elif algorithm == Regularizer.Algorithm.TGV_PD:
            pars['algorithm'] = algorithm
            pars['input'] = None
            pars['first_order_term'] = None
            pars['second_order_term'] = None
            pars['number_of_iterations'] = None
            pars['regularization_parameter'] = None
            
            
            
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
        if None in self.pars:
                raise Exception("Not all parameters have been provided")
                
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
            return self.algorithm(input, 
                              regularization_parameter,
                              self.pars['time_step'] , 
                              self.pars['number_of_iterations'],
                              self.pars['tolerance_constant'],
                              self.pars['restrictive_Z_smoothing'] )
        elif self.algorithm == Regularizer.Algorithm.PatchBased_Regul :
            #LLT_model(np::ndarray input, double d_lambda, double d_tau, int iter, double d_epsil, int switcher)
            # no default
            return self.algorithm(input, regularization_parameter,
                                  self.pars['searching_window_ratio'] , 
                                  self.pars['similarity_window_ratio'] , 
                                  self.pars['PB_filtering_parameter'])
        elif self.algorithm == Regularizer.Algorithm.TGV_PD :
            #LLT_model(np::ndarray input, double d_lambda, double d_tau, int iter, double d_epsil, int switcher)
            # no default
            return self.algorithm(input, regularization_parameter,
                                  self.pars['first_order_term'] , 
                                  self.pars['second_order_term'] , 
                                  self.pars['number_of_iterations'])
            
            
        
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
        reg = Regularizer(Regularizer.Algorithm.LLT_model)
        out = list( reg(input, regularization_parameter, time_step=time_step, 
                        number_of_iterations=number_of_iterations,
                        tolerance_constant=tolerance_constant, 
                        restrictive_Z_smoothing=restrictive_Z_smoothing) )
        out.append(reg.pars)
        return out
    
    @staticmethod
    def PatchBased_Regul(input, regularization_parameter,
                        searching_window_ratio, 
                        similarity_window_ratio,
                        PB_filtering_parameter):
        reg = Regularizer(Regularizer.Algorithm.PatchBased_Regul)   
        out = list( reg(input, 
                        regularization_parameter,
                        searching_window_ratio=searching_window_ratio, 
                        similarity_window_ratio=similarity_window_ratio,
                        PB_filtering_parameter=PB_filtering_parameter )
            )
        out.append(reg.pars)
        return out
    
    @staticmethod
    def TGV_PD(input, regularization_parameter , first_order_term, 
               second_order_term, number_of_iterations):
        
        reg = Regularizer(Regularizer.Algorithm.TGV_PD)
        out = list( reg(input, regularization_parameter, 
                        first_order_term=first_order_term, 
                        second_order_term=second_order_term,
                        number_of_iterations=number_of_iterations) )
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

## plot 
fig = plt.figure()
#a=fig.add_subplot(3,3,1)
#a.set_title('Original')
#imgplot = plt.imshow(Im)

a=fig.add_subplot(2,3,1)
a.set_title('noise')
imgplot = plt.imshow(u0)

reg_output = []
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
reg_output.append(out2)

a=fig.add_subplot(2,3,2)
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
imgplot = plt.imshow(reg_output[-1][0])

###################### FGP_TV #########################################
# u = FGP_TV(single(u0), 0.05, 100, 1e-04);
out2 = Regularizer.FGP_TV(input=u0, regularization_parameter=0.05,
                          number_of_iterations=10)
pars = out2[-1]

reg_output.append(out2)

a=fig.add_subplot(2,3,3)
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
imgplot = plt.imshow(reg_output[-1][0])

###################### LLT_model #########################################
# * u0 = Im + .03*randn(size(Im)); % adding noise
# [Den] = LLT_model(single(u0), 10, 0.1, 1);
#Den = LLT_model(single(u0), 25, 0.0003, 300, 0.0001, 0); 
#input, regularization_parameter , time_step, number_of_iterations,
#                  tolerance_constant, restrictive_Z_smoothing=0
out2 = Regularizer.LLT_model(input=u0, regularization_parameter=25,
                          time_step=0.0003,
                          tolerance_constant=0.0001,
                          number_of_iterations=300)
pars = out2[-1]

reg_output.append(out2)

a=fig.add_subplot(2,3,4)
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
imgplot = plt.imshow(reg_output[-1][0])

###################### PatchBased_Regul #########################################
# Quick 2D denoising example in Matlab:   
#   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
#   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
#   ImDen = PB_Regul_CPU(single(u0), 3, 1, 0.08, 0.05); 

out2 = Regularizer.PatchBased_Regul(input=u0, regularization_parameter=0.05,
                          searching_window_ratio=3,
                          similarity_window_ratio=1,
                          PB_filtering_parameter=0.08)
pars = out2[-1]
reg_output.append(out2)

a=fig.add_subplot(2,3,5)
a.set_title('PatchBased_Regul')
textstr = 'regularization_parameter=%.2f\nsearching_window_ratio=%d\nsimilarity_window_ratio=%.2e\nPB_filtering_parameter=%f'
textstr = textstr % (pars['regularization_parameter'], 
                     pars['searching_window_ratio'], 
                     pars['similarity_window_ratio'],
                     pars['PB_filtering_parameter'])




# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0])


###################### TGV_PD #########################################
# Quick 2D denoising example in Matlab:   
#   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
#   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
#   u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550);


out2 = Regularizer.TGV_PD(input=u0, regularization_parameter=0.05,
                          first_order_term=1.3,
                          second_order_term=1,
                          number_of_iterations=550)
pars = out2[-1]
reg_output.append(out2)

a=fig.add_subplot(2,3,6)
a.set_title('TGV_PD')
textstr = 'regularization_parameter=%.2f\nfirst_order_term=%.2f\nsecond_order_term=%.2f\nnumber_of_iterations=%d'
textstr = textstr % (pars['regularization_parameter'], 
                     pars['first_order_term'], 
                     pars['second_order_term'],
                     pars['number_of_iterations'])




# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0])




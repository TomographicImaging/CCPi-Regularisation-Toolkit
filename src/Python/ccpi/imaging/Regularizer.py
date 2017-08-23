# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:26:00 2017

@author: ofn77899
"""

from ccpi.imaging import cpu_regularizers
import numpy as np
from enum import Enum
import timeit

class Regularizer():
    '''Class to handle regularizer algorithms to be used during reconstruction
    
    Currently 5 CPU (OMP) regularization algorithms are available:
        
    1) SplitBregman_TV
    2) FGP_TV
    3) LLT_model
    4) PatchBased_Regul
    5) TGV_PD
    
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
        SplitBregman_TV = cpu_regularizers.SplitBregman_TV
        FGP_TV = cpu_regularizers.FGP_TV
        LLT_model = cpu_regularizers.LLT_model
        PatchBased_Regul = cpu_regularizers.PatchBased_Regul
        TGV_PD = cpu_regularizers.TGV_PD
    # Algorithm
    
    class TotalVariationPenalty(Enum):
        isotropic = 0
        l1 = 1
    # TotalVariationPenalty
        
    def __init__(self , algorithm, debug = True):
        self.setAlgorithm ( algorithm )
        self.debug = debug
    # __init__
    
    def setAlgorithm(self, algorithm):
        self.algorithm = algorithm
        self.pars = self.getDefaultParsForAlgorithm(algorithm)
    # setAlgorithm
        
    def getDefaultParsForAlgorithm(self, algorithm):
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
            
        else:
            raise Exception('Unknown regularizer algorithm')
            
        return pars
    # parsForAlgorithm
    
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
            if key in self.pars.keys():
                self.pars[key] = value
            else:
                raise Exception('Wrong parameter {0} for regularizer algorithm'.format(key))
    # setParameter
	
    def getParameter(self, **kwargs):
        ret = {}
        for key , value in kwargs.items():
            if key in self.pars.keys():
                ret[key] = self.pars[key]
        else:
            raise Exception('Wrong parameter {0} for regularizer algorithm'.format(key))
    # setParameter
	
        
    def __call__(self, input = None, regularization_parameter = None, **kwargs):
        '''Actual call for the regularizer. 
        
        One can either set the regularization parameters first and then call the
        algorithm or set the regularization parameter during the call (as 
        is done in the static methods). 
        '''
        
        if kwargs is not None:
            for key, value in kwargs.items():
                #print("{0} = {1}".format(key, value))                        
                self.pars[key] = value
                    
        if input is not None: 
            self.pars['input'] = input
        if regularization_parameter is not None:
            self.pars['regularization_parameter'] = regularization_parameter
            
        if self.debug:
            print ("--------------------------------------------------")
            for key, value in self.pars.items():
                if key== 'algorithm' :
                    print("{0} = {1}".format(key, value.__name__))
                elif key == 'input':
                    print("{0} = {1}".format(key, np.shape(value)))
                else:
                    print("{0} = {1}".format(key, value))
        
            
        if None in self.pars:
                raise Exception("Not all parameters have been provided")
        
        input = self.pars['input']
        regularization_parameter = self.pars['regularization_parameter']
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
            if len(np.shape(input)) == 2:
                return self.algorithm(input, regularization_parameter,
                                  self.pars['first_order_term'] , 
                                  self.pars['second_order_term'] , 
                                  self.pars['number_of_iterations'])
            elif len(np.shape(input)) == 3:
                #assuming it's 3D
                # run independent calls on each slice
                out3d = input.copy()
                for i in range(np.shape(input)[2]):
                    out = self.algorithm(input, regularization_parameter,
                                 self.pars['first_order_term'] , 
                                 self.pars['second_order_term'] , 
                                 self.pars['number_of_iterations'])
                    # copy the result in the 3D image
                    out3d.T[i] = out[0].copy()
                # append the rest of the info that the algorithm returns
                output = [out3d]
                for i in range(1,len(out)):
                    output.append(out[i])
                return output
                
                
            
            
        
    # __call__
    
    @staticmethod
    def SplitBregman_TV(input, regularization_parameter , **kwargs):
        start_time = timeit.default_timer()
        reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
        out = list( reg(input, regularization_parameter, **kwargs) )
        out.append(reg.pars)
        txt = reg.printParametersToString()
        txt += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        out.append(txt)
        return out
        
    @staticmethod
    def FGP_TV(input, regularization_parameter , **kwargs):
        start_time = timeit.default_timer()
        reg = Regularizer(Regularizer.Algorithm.FGP_TV)
        out = list( reg(input, regularization_parameter, **kwargs) )
        out.append(reg.pars)
        txt = reg.printParametersToString()
        txt += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        out.append(txt)
        return out
    
    @staticmethod
    def LLT_model(input, regularization_parameter , time_step, number_of_iterations,
                  tolerance_constant, restrictive_Z_smoothing=0):
        start_time = timeit.default_timer()
        reg = Regularizer(Regularizer.Algorithm.LLT_model)
        out = list( reg(input, regularization_parameter, time_step=time_step, 
                        number_of_iterations=number_of_iterations,
                        tolerance_constant=tolerance_constant, 
                        restrictive_Z_smoothing=restrictive_Z_smoothing) )
        out.append(reg.pars)
        txt = reg.printParametersToString()
        txt += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        out.append(txt)
        return out
    
    @staticmethod
    def PatchBased_Regul(input, regularization_parameter,
                        searching_window_ratio, 
                        similarity_window_ratio,
                        PB_filtering_parameter):
        start_time = timeit.default_timer()
        reg = Regularizer(Regularizer.Algorithm.PatchBased_Regul)   
        out = list( reg(input, 
                        regularization_parameter,
                        searching_window_ratio=searching_window_ratio, 
                        similarity_window_ratio=similarity_window_ratio,
                        PB_filtering_parameter=PB_filtering_parameter )
            )
        out.append(reg.pars)
        txt = reg.printParametersToString()
        txt += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        out.append(txt)
        return out
    
    @staticmethod
    def TGV_PD(input, regularization_parameter , first_order_term, 
               second_order_term, number_of_iterations):
        start_time = timeit.default_timer()
        
        reg = Regularizer(Regularizer.Algorithm.TGV_PD)
        out = list( reg(input, regularization_parameter, 
                        first_order_term=first_order_term, 
                        second_order_term=second_order_term,
                        number_of_iterations=number_of_iterations) )
        out.append(reg.pars)
        txt = reg.printParametersToString()
        txt += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        out.append(txt)
        
        return out
    
    def printParametersToString(self):
        txt = r''
        for key, value in self.pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
        

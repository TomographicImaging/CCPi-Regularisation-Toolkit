import unittest
import numpy as np
import os
from ccpi.filters.regularisers import ROF_TV, FGP_TV
from qualitymetrics import rmse
import matplotlib.pyplot as plt

class TestRegularisers(unittest.TestCase):
    def __init__(self):
        filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")
        
        # read noiseless image
        Im = plt.imread(filename)
        Im = np.asarray(Im, dtype='float32')

        Im = Im/255
        self.u0 = Im
        self.Im = Im
        self.tolerance = 0.00001
        self.rms_rof_exp = 0.01 #expected value for ROF model
        self.rms_fgp_exp = 0.01 #expected value for FGP model
        
        # set parameters for ROF-TV
        self.pars_rof_tv = {'algorithm': ROF_TV, \
        'input' : self.u0,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 50,\
        'time_marching_parameter': 0.0025
        }
        # set parameters for FGP-TV
        self.pars_fgp_tv = {'algorithm' : FGP_TV, \
        'input' : self.u0,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :50 ,\
        'tolerance_constant':0.00001,\
        'methodTV': 0 ,\
        'nonneg': 0 ,\
        'printingOut': 0 
        }
    def test_cpu_regularisers(self):
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing ROF-TV (2D, CPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        rof_cpu = ROF_TV(self.pars_rof_tv['input'],
             self.pars_rof_tv['regularisation_parameter'],
             self.pars_rof_tv['number_of_iterations'],
             self.pars_rof_tv['time_marching_parameter'],'cpu')
        rms_rof = rmse(self.Im, rof_cpu)
        # now compare obtained rms with the expected value
        if abs(rms_rof-self.rms_rof_exp) > self.tolerance:
            raise TypeError('ROF-TV (2D, CPU) test FAILED')
        else:
            print ("test PASSED")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing FGP-TV (2D, CPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        fgp_cpu = FGP_TV(self.pars_fgp_tv['input'], 
              self.pars_fgp_tv['regularisation_parameter'],
              self.pars_fgp_tv['number_of_iterations'],
              self.pars_fgp_tv['tolerance_constant'], 
              self.pars_fgp_tv['methodTV'],
              self.pars_fgp_tv['nonneg'],
              self.pars_fgp_tv['printingOut'],'cpu')  
        rms_fgp = rmse(self.Im, fgp_cpu)
        # now compare obtained rms with the expected value
        if abs(rms_fgp-self.rms_fgp_exp) > self.tolerance:
            raise TypeError('FGP-TV (2D, CPU) test FAILED')
        else:
            print ("test PASSED")
    def test_gpu_regularisers(self):
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing ROF-TV (2D, GPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        rof_gpu = ROF_TV(self.pars_rof_tv['input'],
             self.pars_rof_tv['regularisation_parameter'],
             self.pars_rof_tv['number_of_iterations'],
             self.pars_rof_tv['time_marching_parameter'],'gpu')
        rms_rof = rmse(self.Im, rof_gpu)
        # now compare obtained rms with the expected value
        if abs(rms_rof-self.rms_rof_exp) > self.tolerance:
            raise TypeError('ROF-TV (2D, GPU) test FAILED')
        else:
            print ("test PASSED")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("_________testing FGP-TV (2D, GPU)__________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        fgp_gpu = FGP_TV(self.pars_fgp_tv['input'], 
              self.pars_fgp_tv['regularisation_parameter'],
              self.pars_fgp_tv['number_of_iterations'],
              self.pars_fgp_tv['tolerance_constant'], 
              self.pars_fgp_tv['methodTV'],
              self.pars_fgp_tv['nonneg'],
              self.pars_fgp_tv['printingOut'],'gpu')  
        rms_fgp = rmse(self.Im, fgp_gpu)
        if abs(rms_fgp-self.rms_fgp_exp) > self.tolerance:
            raise TypeError('FGP-TV (2D, GPU) test FAILED')
        else:
            print ("test PASSED")
        # now compare obtained rms with the expected value
        self.assertLess(...)
if __name__ == "__main__":
    unittest.main()
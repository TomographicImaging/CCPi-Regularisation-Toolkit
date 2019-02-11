import unittest
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from PIL import Image

class TestTVROF(unittest.TestCase):
    
    def test_ROF_TV_CPU_vs_GPU(self):
        #print ("tomas debug test function")
        print(__name__)
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
        # read image
        Im = plt.imread(filename)                     
        Im = np.asarray(Im, dtype='float32')
        
        Im = Im/255
        perc = 0.05
        u0 = Im + np.random.normal(loc = 0 ,
                                          scale = perc * Im , 
                                          size = np.shape(Im))
        u_ref = Im + np.random.normal(loc = 0 ,
                                          scale = 0.01 * Im , 
                                          size = np.shape(Im))
        
        # map the u0 u0->u0>0
        # f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
        u0 = u0.astype('float32')
        u_ref = u_ref.astype('float32')
        
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print ("____________ROF-TV bench___________________")
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        # set parameters
        pars = {'algorithm': ROF_TV, \
        'input' : u0,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 2500,\
        'time_marching_parameter': 0.00002
        }
        print ("#############ROF TV CPU####################")
        start_time = timeit.default_timer()
        rof_cpu = ROF_TV(pars['input'],
                     pars['regularisation_parameter'],
                     pars['number_of_iterations'],
                     pars['time_marching_parameter'],'cpu')
        rms = rmse(Im, rof_cpu)
        pars['rmse'] = rms
        
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("##############ROF TV GPU##################")
        start_time = timeit.default_timer()
        try:
            rof_gpu = ROF_TV(pars['input'], 
                             pars['regularisation_parameter'],
                             pars['number_of_iterations'], 
                             pars['time_marching_parameter'],'gpu')
        except ValueError as ve:
            self.skipTest("Results not comparable. GPU computing error.")

        rms = rmse(Im, rof_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = ROF_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
        print (txtstr)
        print ("--------Compare the results--------")
        tolerance = 1e-04
        diff_im = np.zeros(np.shape(rof_cpu))
        diff_im = abs(rof_cpu - rof_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum() , 1)


if __name__ == '__main__':
    unittest.main()
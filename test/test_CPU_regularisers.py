import unittest
#import math
import os
#import timeit
import numpy as np
from ccpi.filters.regularisers import FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th, ROF_TV, PD_TV
from testroutines import BinReader, rmse 
###############################################################################

class TestRegularisers(unittest.TestCase):

    def getPars(self):
        #filename = os.path.join("test","lena_gray_512.tif")
        #plt = TiffReader()
        filename = os.path.join("test","test_imageLena.bin")
        plt = BinReader()
        # read image
        Im = plt.imread(filename)
        Im = np.asarray(Im, dtype='float32')
        Im = Im / 255
        perc = 0.05
        u0 = Im + np.random.normal(loc=0,
                                   scale=perc * Im,
                                   size=np.shape(Im))
        u_ref = Im + np.random.normal(loc=0,
                                      scale=0.01 * Im,
                                      size=np.shape(Im))
        u0 = u0.astype('float32')
        u_ref = u_ref.astype('float32')
        return Im,u0,u_ref


    def test_FGP_TV_CPU(self):
        Im,input,ref = self.getPars()

        fgp_cpu,info = FGP_TV(input,0.02,300,0.0,0,0,'cpu');

        rms = rmse(Im, fgp_cpu)

        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_PD_TV_CPU(self):
        Im,input,ref = self.getPars()

        pd_cpu,info = PD_TV(input, 0.02, 300, 0.0, 0, 0, 8, 0.0025, 'cpu');

        rms = rmse(Im, pd_cpu)
        
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_TV_ROF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        fgp_cpu,info = ROF_TV(input,0.02,1000,0.001,0.0, 'cpu')

        rms = rmse(Im, fgp_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_SB_TV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu,info = SB_TV(input,0.02,150,0.0,0,'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_TGV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        tgv_cpu,info = TGV(input,0.02,1.0,2.0,500,12,0.0,'cpu')

        rms = rmse(Im, tgv_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_LLT_ROF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu,info = LLT_ROF(input,0.01,0.008,1000,0.001,0.0,'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_NDF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu,info = NDF(input, 0.02, 0.17,1000,0.01,1,0.0, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

    def test_Diff4th_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu,info = Diff4th(input, 0.8,0.02,1000,0.001,0.0, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

    def test_FGP_dTV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu,info = FGP_dTV(input,ref,0.02,500,0.0,0.2,0,0, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

if __name__ == '__main__':
    unittest.main()

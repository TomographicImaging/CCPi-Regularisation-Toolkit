import unittest
import math
import os
import timeit
import numpy as np
from ccpi.filters.regularisers import FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th, ROF_TV
from testroutines import *

###############################################################################

class TestRegularisers(unittest.TestCase):

    def getPars(self):
        filename = os.path.join("lena_gray_512.tif")
        plt = TiffReader()
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

        fgp_cpu = FGP_TV(input,0.04,1200,1e-5,0,0,0,'cpu');

        rms = rmse(Im, fgp_cpu)

        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_TV_ROF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        fgp_cpu = ROF_TV(input,0.04,1200,2e-5, 'cpu')

        rms = rmse(Im, fgp_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_SB_TV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = SB_TV(input,0.04,150,1e-5,0,0,'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_TGV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = TGV(input,0.04,1.0,2.0,250,12,'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_LLT_ROF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = LLT_ROF(input,0.04,0.01,1000,1e-4,'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_NDF_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = NDF(input, 0.06, 0.04,1000,0.025,1, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

    def test_Diff4th_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = Diff4th(input, 3.5,0.02,500,0.001, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

    def test_FGP_dTV_CPU(self):
        # set parameters
        Im, input,ref = self.getPars()
        # call routine
        sb_cpu = FGP_dTV(input,ref,0.04,1000,1e-7,0.2,0,0,0, 'cpu')

        rms = rmse(Im, sb_cpu)

        # now test that it generates some expected output
        self.assertAlmostEqual(rms, 0.02, delta=0.01)

if __name__ == '__main__':
    unittest.main()

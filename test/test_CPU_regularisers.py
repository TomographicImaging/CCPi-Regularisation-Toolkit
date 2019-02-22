import unittest
import math
import os
import timeit
from ccpi.filters.regularisers import FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th, ROF_TV
from testroutines import *

###############################################################################

class TestRegularisers(unittest.TestCase):

    def getPars(self,alg,noi=1200):
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
        # map the u0 u0->u0>0
        # f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
        u0 = u0.astype('float32')
        u_ref = u_ref.astype('float32')
        # set parameters
        pars = {'algorithm': alg, \
                'input': u0, \
                'regularisation_parameter': 0.04, \
                'number_of_iterations': noi, \
                'tolerance_constant': 0.00001, \
                'methodTV': 0, \
                'nonneg': 0, \
                'printingOut': 0, \
                'time_marching_parameter': 0.00002
                }
        return Im, pars


    def test_FGP_TV_CPU(self):
        Im, pars = self.getPars(FGP_TV)

        fgp_cpu = FGP_TV(pars['input'],
                         pars['regularisation_parameter'],
                         pars['number_of_iterations'],
                         pars['tolerance_constant'],
                         pars['methodTV'],
                         pars['nonneg'],
                         pars['printingOut'], 'cpu')

        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_TV_ROF_CPU(self):
        # set parameters
        Im, pars = self.getPars(ROF_TV)
        # call routine
        fgp_cpu = ROF_TV(pars['input'],
                         pars['regularisation_parameter'],
                         pars['number_of_iterations'],
                         pars['time_marching_parameter'], 'cpu')

        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms

        #txtstr = printParametersToString(pars)
        #print(txtstr)
        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

    def test_SB_TV_CPU(self):
        # set parameters
        Im, pars = self.getPars(SB_TV)
        # call routine
        fgp_cpu = SB_TV(pars['input'],
                         pars['regularisation_parameter'],
                         pars['number_of_iterations'],
                         pars['time_marching_parameter'], 'cpu')

        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms

        #txtstr = printParametersToString(pars)
        #print(txtstr)
        # now test that it generates some expected output
        self.assertAlmostEqual(rms,0.02,delta=0.01)

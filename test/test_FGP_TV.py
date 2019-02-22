import unittest
import math
import os
import timeit
from ccpi.filters.regularisers import FGP_TV
#, FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from testroutines import *

###############################################################################

class TestRegularisers(unittest.TestCase):

    def test_FGP_TV_CPU(self):
        print(__name__)
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

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("____________FGP-TV bench___________________")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # set parameters
        pars = {'algorithm': FGP_TV, \
                'input': u0, \
                'regularisation_parameter': 0.04, \
                'number_of_iterations': 1200, \
                'tolerance_constant': 0.00001, \
                'methodTV': 0, \
                'nonneg': 0, \
                'printingOut': 0
                }

        print("#############FGP TV CPU####################")
        start_time = timeit.default_timer()
        fgp_cpu = FGP_TV(pars['input'],
                         pars['regularisation_parameter'],
                         pars['number_of_iterations'],
                         pars['tolerance_constant'],
                         pars['methodTV'],
                         pars['nonneg'],
                         pars['printingOut'], 'cpu')

        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms

        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time', timeit.default_timer() - start_time)
        print(txtstr)
        self.assertTrue(math.isclose(rms,0.02,rel_tol=1e-1))

    def test_FGP_TV_CPU_vs_GPU(self):
        print(__name__)
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

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("____________FGP-TV bench___________________")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # set parameters
        pars = {'algorithm': FGP_TV, \
                'input': u0, \
                'regularisation_parameter': 0.04, \
                'number_of_iterations': 1200, \
                'tolerance_constant': 0.00001, \
                'methodTV': 0, \
                'nonneg': 0, \
                'printingOut': 0
                }

        print("#############FGP TV CPU####################")
        start_time = timeit.default_timer()
        fgp_cpu = FGP_TV(pars['input'],
                         pars['regularisation_parameter'],
                         pars['number_of_iterations'],
                         pars['tolerance_constant'],
                         pars['methodTV'],
                         pars['nonneg'],
                         pars['printingOut'], 'cpu')

        rms = rmse(Im, fgp_cpu)
        pars['rmse'] = rms

        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time', timeit.default_timer() - start_time)
        print(txtstr)

        print("##############FGP TV GPU##################")
        start_time = timeit.default_timer()
        try:
            fgp_gpu = FGP_TV(pars['input'],
                             pars['regularisation_parameter'],
                             pars['number_of_iterations'],
                             pars['tolerance_constant'],
                             pars['methodTV'],
                             pars['nonneg'],
                             pars['printingOut'], 'gpu')

        except ValueError as ve:
            self.skipTest("Results not comparable. GPU computing error.")

        rms = rmse(Im, fgp_gpu)
        pars['rmse'] = rms
        pars['algorithm'] = FGP_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ('elapsed time', timeit.default_timer() - start_time)
        print(txtstr)

        print("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(fgp_cpu))
        diff_im = abs(fgp_cpu - fgp_gpu)
        diff_im[diff_im > tolerance] = 1

        self.assertLessEqual(diff_im.sum(), 1)

if __name__ == '__main__':
    unittest.main()

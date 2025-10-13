import unittest
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import (
    ROF_TV,
    FGP_TV,
    PD_TV,
    SB_TV,
    TGV,
    LLT_ROF,
    FGP_dTV,
    NDF,
    Diff4th,
)
from ccpi.filters.utils import cilregcuda

gpu_modules_available = cilregcuda is not None
from testroutines import BinReader, rmse, printParametersToString


@unittest.skipUnless(gpu_modules_available, "Skipping as GPU modules not available")
class TestRegularisers(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(os.path.dirname(__file__), "test_imageLena.bin")
        # lena_gray_512.tif

    def _initiate_data(self):
        plt = BinReader()
        # read image
        Im = plt.imread(self.filename)
        Im = np.asarray(Im, dtype="float32")

        Im = Im / 255
        perc = 0.05
        slices = 20

        noisyVol = np.zeros((slices, 512, 512), dtype="float32")
        noisyRef = np.zeros((slices, 512, 512), dtype="float32")
        idealVol = np.zeros((slices, 512, 512), dtype="float32")

        for i in range(slices):
            noisyVol[i, :, :] = Im + np.random.normal(
                loc=0, scale=perc * Im, size=np.shape(Im)
            )
            noisyRef[i, :, :] = Im + np.random.normal(
                loc=0, scale=0.01 * Im, size=np.shape(Im)
            )
            idealVol[i, :, :] = Im
        return noisyVol, noisyRef, idealVol

    def test_ROF_TV_CPU_vs_GPU(self):

        noisyVol, noisyRef, idealVol = self._initiate_data()

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("____________ROF-TV bench___________________")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # set parameters
        pars = {
            "algorithm": ROF_TV,
            "input": noisyVol,
            "regularisation_parameter": 0.02,
            "number_of_iterations": 100,
            "time_marching_parameter": 0.001,
            "tolerance_constant": 0.0,
        }
        print("#############ROF TV CPU####################")
        start_time = timeit.default_timer()
        rof_cpu = ROF_TV(
            pars["input"],
            pars["regularisation_parameter"],
            pars["number_of_iterations"],
            pars["time_marching_parameter"],
            pars["tolerance_constant"],
            device="cpu",
        )
        rms = rmse(idealVol, rof_cpu)
        pars["rmse"] = rms

        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
        print(txtstr)
        print("##############ROF TV GPU##################")
        start_time = timeit.default_timer()
        rof_gpu = ROF_TV(
            pars["input"],
            pars["regularisation_parameter"],
            pars["number_of_iterations"],
            pars["time_marching_parameter"],
            pars["tolerance_constant"],
            device="gpu",
        )

        rms = rmse(idealVol, rof_gpu)
        pars["rmse"] = rms
        pars["algorithm"] = ROF_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
        print(txtstr)
        print("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = np.zeros(np.shape(rof_cpu))
        diff_im = abs(rof_cpu - rof_gpu)
        diff_im[diff_im > tolerance] = 1
        self.assertLessEqual(diff_im.sum(), 1)

    def test_FGP_TV_CPU_vs_GPU(self):
        noisyVol, noisyRef, idealVol = self._initiate_data()

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("____________FGP-TV bench___________________")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # set parameters
        pars = {
            "algorithm": FGP_TV,
            "input": noisyVol,
            "regularisation_parameter": 0.05,
            "number_of_iterations": 200,
            "tolerance_constant": 0.0,
            "methodTV": 0,
            "nonneg": 0,
        }

        print("#############FGP TV CPU####################")
        start_time = timeit.default_timer()
        fgp_cpu = FGP_TV(
            pars["input"],
            pars["regularisation_parameter"],
            pars["number_of_iterations"],
            pars["tolerance_constant"],
            pars["methodTV"],
            pars["nonneg"],
            device="cpu",
        )

        rms = rmse(idealVol, fgp_cpu)
        pars["rmse"] = rms

        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
        print(txtstr)

        print("##############FGP TV GPU##################")
        start_time = timeit.default_timer()
        fgp_gpu = FGP_TV(
            pars["input"],
            pars["regularisation_parameter"],
            pars["number_of_iterations"],
            pars["tolerance_constant"],
            pars["methodTV"],
            pars["nonneg"],
            device="gpu",
        )

        rms = rmse(idealVol, fgp_gpu)
        pars["rmse"] = rms
        pars["algorithm"] = FGP_TV
        txtstr = printParametersToString(pars)
        txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
        print(txtstr)

        print("--------Compare the results--------")
        tolerance = 1e-05
        diff_im = abs(fgp_cpu - fgp_gpu)
        np.testing.assert_array_less(diff_im, tolerance)

if __name__ == "__main__":
    unittest.main()

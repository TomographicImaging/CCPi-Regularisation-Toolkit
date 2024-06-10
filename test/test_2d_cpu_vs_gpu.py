import pytest
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
from conftest import rmse, printParametersToString
from numpy.testing import assert_allclose


def test_ROF_TV_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    # set parameters
    pars = {
        "algorithm": ROF_TV,
        "input": host_pepper_im_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 1000,
        "time_marching_parameter": 0.001,
        "tolerance_constant": 0.0,
    }
    print("#############ROF TV CPU####################")
    rof_cpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, rof_cpu)
    print("##############ROF TV GPU##################")
    rof_gpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, rof_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(rof_cpu), np.max(rof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert rof_cpu.dtype == np.float32
    assert rof_gpu.dtype == np.float32


def test_ROF_TV_CPU_vs_GPU_nonsquare(
    host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
):
    # set parameters
    pars = {
        "algorithm": ROF_TV,
        "input": host_pepper_im_noise_nonsquare,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 1000,
        "time_marching_parameter": 0.001,
        "tolerance_constant": 0.0,
    }
    print("#############ROF TV CPU####################")
    rof_cpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im_nonsquare, rof_cpu)
    print("##############ROF TV GPU##################")
    rof_gpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im_nonsquare, rof_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(rof_cpu), np.max(rof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert rof_cpu.dtype == np.float32
    assert rof_gpu.dtype == np.float32


def test_FGP_TV_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    pars = {
        "algorithm": FGP_TV,
        "input": host_pepper_im_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 400,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
    }

    print("#############FGP TV CPU####################")
    fgp_cpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, fgp_cpu)
    print("##############FGP TV GPU##################")
    fgp_gpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, fgp_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_cpu), np.max(fgp_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_cpu.dtype == np.float32
    assert fgp_gpu.dtype == np.float32


def test_FGP_TV_CPU_vs_GPU_nonsquare(
    host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
):
    pars = {
        "algorithm": FGP_TV,
        "input": host_pepper_im_noise_nonsquare,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 400,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
    }

    print("#############FGP TV CPU####################")
    fgp_cpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im_nonsquare, fgp_cpu)
    print("##############FGP TV GPU##################")
    fgp_gpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im_nonsquare, fgp_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_cpu), np.max(fgp_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_cpu.dtype == np.float32
    assert fgp_gpu.dtype == np.float32


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_PD_TV_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    pars = {
        "algorithm": PD_TV,
        "input": host_pepper_im_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 1500,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
        "lipschitz_const": 8,
    }

    print("#############PD TV CPU####################")
    pd_cpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, pd_cpu)
    print("##############PD TV GPU##################")
    pd_gpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, pd_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(pd_cpu), np.max(pd_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert pd_cpu.dtype == np.float32
    assert pd_gpu.dtype == np.float32


def test_PD_TV_CPU_vs_GPU_nonsqure(
    host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
):
    pars = {
        "algorithm": PD_TV,
        "input": host_pepper_im_noise_nonsquare,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 1500,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
        "lipschitz_const": 8,
    }

    print("#############PD TV CPU####################")
    pd_cpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im_nonsquare, pd_cpu)
    print("##############PD TV GPU##################")
    pd_gpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im_nonsquare, pd_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(pd_cpu), np.max(pd_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert pd_cpu.dtype == np.float32
    assert pd_gpu.dtype == np.float32


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_SB_TV_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    pars = {
        "algorithm": SB_TV,
        "input": host_pepper_im_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 250,
        "tolerance_constant": 0.0,
        "methodTV": 0,
    }
    print("#############SB TV CPU####################")
    sb_cpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, sb_cpu)
    print("##############SB TV GPU##################")
    sb_gpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, sb_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(sb_cpu), np.max(sb_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert sb_cpu.dtype == np.float32
    assert sb_gpu.dtype == np.float32


def test_SB_TV_CPU_vs_GPU_nonsquare(
    host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
):
    pars = {
        "algorithm": SB_TV,
        "input": host_pepper_im_noise_nonsquare,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 250,
        "tolerance_constant": 0.0,
        "methodTV": 0,
    }
    print("#############SB TV CPU####################")
    sb_cpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im_nonsquare, sb_cpu)
    print("##############SB TV GPU##################")
    sb_gpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im_nonsquare, sb_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(sb_cpu), np.max(sb_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert sb_cpu.dtype == np.float32
    assert sb_gpu.dtype == np.float32


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_TGV_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    pars = {
        "algorithm": TGV,
        "input": host_pepper_im_noise,
        "regularisation_parameter": 0.02,
        "alpha1": 1.0,
        "alpha0": 2.0,
        "number_of_iterations": 1000,
        "LipshitzConstant": 12,
        "tolerance_constant": 0.0,
    }
    print("#############TGV CPU####################")
    tgv_cpu = TGV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["alpha1"],
        pars["alpha0"],
        pars["number_of_iterations"],
        pars["LipshitzConstant"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, tgv_cpu)
    print("##############TGV GPU##################")
    tgv_gpu = TGV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["alpha1"],
        pars["alpha0"],
        pars["number_of_iterations"],
        pars["LipshitzConstant"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, tgv_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(tgv_cpu), np.max(tgv_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert tgv_cpu.dtype == np.float32
    assert tgv_gpu.dtype == np.float32


# TODO: This test fails! A bug in TGV.
# def test_TGV_CPU_vs_GPU_nonsquare(
#     host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
# ):
#     pars = {
#         "algorithm": TGV,
#         "input": host_pepper_im_noise_nonsquare,
#         "regularisation_parameter": 0.02,
#         "alpha1": 1.0,
#         "alpha0": 2.0,
#         "number_of_iterations": 1000,
#         "LipshitzConstant": 12,
#         "tolerance_constant": 0.0,
#     }
#     print("#############TGV CPU####################")
#     tgv_cpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="cpu",
#     )
#     rms_cpu = rmse(host_pepper_im_nonsquare, tgv_cpu)
#     print("##############TGV GPU##################")
#     tgv_gpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="gpu",
#     )
#     rms_gpu = rmse(host_pepper_im_nonsquare, tgv_gpu)

#     print("--------Compare the results--------")
#     eps = 1e-5
#     assert_allclose(rms_cpu, rms_gpu, rtol=eps)
#     assert_allclose(np.max(tgv_cpu), np.max(tgv_gpu), rtol=eps)
#     assert rms_cpu > 0.0
#     assert rms_gpu > 0.0
#     assert tgv_cpu.dtype == np.float32
#     assert tgv_gpu.dtype == np.float32

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_LLT_ROF_CPU_vs_GPU(host_pepper_im, host_pepper_im_noise):
    pars = {
        "algorithm": LLT_ROF,
        "input": host_pepper_im_noise,
        "regularisation_parameterROF": 0.01,
        "regularisation_parameterLLT": 0.0085,
        "number_of_iterations": 1000,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############LLT_ROF CPU####################")
    lltrof_cpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im, lltrof_cpu)
    print("##############LLT_ROF GPU##################")
    lltrof_gpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im, lltrof_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(lltrof_cpu), np.max(lltrof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert lltrof_cpu.dtype == np.float32
    assert lltrof_gpu.dtype == np.float32


def test_LLT_ROF_CPU_vs_GPU_nonsquare(
    host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
):
    pars = {
        "algorithm": LLT_ROF,
        "input": host_pepper_im_noise_nonsquare,
        "regularisation_parameterROF": 0.01,
        "regularisation_parameterLLT": 0.0085,
        "number_of_iterations": 1000,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############LLT_ROF CPU####################")
    lltrof_cpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_im_nonsquare, lltrof_cpu)
    print("##############LLT_ROF GPU##################")
    lltrof_gpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_im_nonsquare, lltrof_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(lltrof_cpu), np.max(lltrof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert lltrof_cpu.dtype == np.float32
    assert lltrof_gpu.dtype == np.float32


#     def test_LLT_ROF_CPU_vs_GPU(self):
#         u0, u_ref, Im = self._initiate_data()

#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("____________LLT-ROF bench___________________")
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#         # set parameters
#         pars = {
#             "algorithm": LLT_ROF,
#             "input": u0,
#             "regularisation_parameterROF": 0.01,
#             "regularisation_parameterLLT": 0.0085,
#             "number_of_iterations": 1000,
#             "time_marching_parameter": 0.0001,
#             "tolerance_constant": 0.0,
#         }

#         print("#############LLT- ROF CPU####################")
#         start_time = timeit.default_timer()
#         lltrof_cpu = LLT_ROF(
#             pars["input"],
#             pars["regularisation_parameterROF"],
#             pars["regularisation_parameterLLT"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["tolerance_constant"],
#             device="cpu",
#         )

#         rms = rmse(Im, lltrof_cpu)
#         pars["rmse"] = rms

#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("#############LLT- ROF GPU####################")
#         start_time = timeit.default_timer()
#         lltrof_gpu = LLT_ROF(
#             pars["input"],
#             pars["regularisation_parameterROF"],
#             pars["regularisation_parameterLLT"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["tolerance_constant"],
#             device="gpu",
#         )

#         rms = rmse(Im, lltrof_gpu)
#         pars["rmse"] = rms
#         pars["algorithm"] = LLT_ROF
#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("--------Compare the results--------")
#         tolerance = 1e-05
#         diff_im = np.zeros(np.shape(lltrof_gpu))
#         diff_im = abs(lltrof_cpu - lltrof_gpu)
#         diff_im[diff_im > tolerance] = 1
#         self.assertLessEqual(diff_im.sum(), 1)

#     def test_NDF_CPU_vs_GPU(self):
#         u0, u_ref, Im = self._initiate_data()

#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("_______________NDF bench___________________")
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#         # set parameters
#         pars = {
#             "algorithm": NDF,
#             "input": u0,
#             "regularisation_parameter": 0.02,
#             "edge_parameter": 0.017,
#             "number_of_iterations": 1500,
#             "time_marching_parameter": 0.01,
#             "penalty_type": 1,
#             "tolerance_constant": 0.0,
#         }

#         print("#############NDF CPU####################")
#         start_time = timeit.default_timer()
#         ndf_cpu = NDF(
#             pars["input"],
#             pars["regularisation_parameter"],
#             pars["edge_parameter"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["penalty_type"],
#             pars["tolerance_constant"],
#             device="cpu",
#         )

#         rms = rmse(Im, ndf_cpu)
#         pars["rmse"] = rms

#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)

#         print("##############NDF GPU##################")
#         start_time = timeit.default_timer()
#         ndf_gpu = NDF(
#             pars["input"],
#             pars["regularisation_parameter"],
#             pars["edge_parameter"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["penalty_type"],
#             pars["tolerance_constant"],
#             device="gpu",
#         )

#         rms = rmse(Im, ndf_gpu)
#         pars["rmse"] = rms
#         pars["algorithm"] = NDF
#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("--------Compare the results--------")
#         tolerance = 1e-05
#         diff_im = np.zeros(np.shape(ndf_cpu))
#         diff_im = abs(ndf_cpu - ndf_gpu)
#         diff_im[diff_im > tolerance] = 1
#         self.assertLessEqual(diff_im.sum(), 1)

#     def test_Diff4th_CPU_vs_GPU(self):
#         u0, u_ref, Im = self._initiate_data()

#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("___Anisotropic Diffusion 4th Order (2D)____")
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#         # set parameters
#         pars = {
#             "algorithm": Diff4th,
#             "input": u0,
#             "regularisation_parameter": 0.8,
#             "edge_parameter": 0.02,
#             "number_of_iterations": 1000,
#             "time_marching_parameter": 0.0001,
#             "tolerance_constant": 0.0,
#         }

#         print("#############Diff4th CPU####################")
#         start_time = timeit.default_timer()
#         diff4th_cpu = Diff4th(
#             pars["input"],
#             pars["regularisation_parameter"],
#             pars["edge_parameter"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["tolerance_constant"],
#             device="cpu",
#         )

#         rms = rmse(Im, diff4th_cpu)
#         pars["rmse"] = rms

#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("##############Diff4th GPU##################")
#         start_time = timeit.default_timer()
#         diff4th_gpu = Diff4th(
#             pars["input"],
#             pars["regularisation_parameter"],
#             pars["edge_parameter"],
#             pars["number_of_iterations"],
#             pars["time_marching_parameter"],
#             pars["tolerance_constant"],
#             device="gpu",
#         )

#         rms = rmse(Im, diff4th_gpu)
#         pars["rmse"] = rms
#         pars["algorithm"] = Diff4th
#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("--------Compare the results--------")
#         tolerance = 1e-05
#         diff_im = np.zeros(np.shape(diff4th_cpu))
#         diff_im = abs(diff4th_cpu - diff4th_gpu)
#         diff_im[diff_im > tolerance] = 1
#         self.assertLessEqual(diff_im.sum(), 1)

#     def test_FDGdTV_CPU_vs_GPU(self):
#         u0, u_ref, Im = self._initiate_data()

#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("____________FGP-dTV bench___________________")
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#         # set parameters
#         pars = {
#             "algorithm": FGP_dTV,
#             "input": u0,
#             "refdata": u_ref,
#             "regularisation_parameter": 0.02,
#             "number_of_iterations": 500,
#             "tolerance_constant": 0.0,
#             "eta_const": 0.2,
#             "methodTV": 0,
#             "nonneg": 0,
#         }

#         print("#############FGP dTV CPU####################")
#         start_time = timeit.default_timer()
#         fgp_dtv_cpu = FGP_dTV(
#             pars["input"],
#             pars["refdata"],
#             pars["regularisation_parameter"],
#             pars["number_of_iterations"],
#             pars["tolerance_constant"],
#             pars["eta_const"],
#             pars["methodTV"],
#             pars["nonneg"],
#             device="cpu",
#         )

#         rms = rmse(Im, fgp_dtv_cpu)
#         pars["rmse"] = rms

#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("##############FGP dTV GPU##################")
#         start_time = timeit.default_timer()
#         fgp_dtv_gpu = FGP_dTV(
#             pars["input"],
#             pars["refdata"],
#             pars["regularisation_parameter"],
#             pars["number_of_iterations"],
#             pars["tolerance_constant"],
#             pars["eta_const"],
#             pars["methodTV"],
#             pars["nonneg"],
#             device="gpu",
#         )

#         rms = rmse(Im, fgp_dtv_gpu)
#         pars["rmse"] = rms
#         pars["algorithm"] = FGP_dTV
#         txtstr = printParametersToString(pars)
#         txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
#         print(txtstr)
#         print("--------Compare the results--------")
#         tolerance = 1e-05
#         diff_im = np.zeros(np.shape(fgp_dtv_cpu))
#         diff_im = abs(fgp_dtv_cpu - fgp_dtv_gpu)
#         diff_im[diff_im > tolerance] = 1
#         self.assertLessEqual(diff_im.sum(), 1)


# if __name__ == "__main__":
#     unittest.main()

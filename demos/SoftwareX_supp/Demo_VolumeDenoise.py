#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This demo scripts support the following publication: 
"CCPi-Regularisation Toolkit for computed tomographic image reconstruction with 
proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Martin J. Turner,
 Philip J. Withers; Software X, 2019
____________________________________________________________________________
* Generates phantom using TomoPhantom software 
* Denoise using closely to optimal parameters
____________________________________________________________________________
>>>>> Dependencies: <<<<<
1. TomoPhantom software for phantom and data generation

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
Apache 2.0.
"""
import timeit
import matplotlib.pyplot as plt

# import matplotlib.gridspec as gridspec
import numpy as np
import os
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.artifacts import ArtifactsClass
from ccpi.supp.qualitymetrics import QualityTools
from scipy.signal import gaussian
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th

# %%
print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 16  # select a model number from the library
N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

# adding normally distributed noise
artifacts_add = ArtifactsClass(phantom_tm)
phantom_noise = artifacts_add.noise(sigma=0.1, noisetype="Gaussian")

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_noise[sliceSel, :, :], vmin=0, vmax=1.4)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_noise[:, sliceSel, :], vmin=0, vmax=1.4)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_noise[:, :, sliceSel], vmin=0, vmax=1.4)
plt.title("3D Phantom, sagittal view")
plt.show()
# %%
print("____________________Applying regularisers_______________________")
print("________________________________________________________________")

print("#############ROF TV CPU####################")
# set parameters
pars = {
    "algorithm": ROF_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 1000,
    "time_marching_parameter": 0.00025,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(rof_cpu3D, infcpu) = ROF_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "cpu",
)

toc = timeit.default_timer()

Run_time_rof = toc - tic
Qtools = QualityTools(phantom_tm, rof_cpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, rof_cpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_rof = Qtools.ssim(win2d)

print(
    "ROF-TV (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE_rof, ssim_rof[0], Run_time_rof
    )
)
# %%
print("#############ROF TV GPU####################")
# set parameters
pars = {
    "algorithm": ROF_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 8330,
    "time_marching_parameter": 0.00025,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(rof_gpu3D, infogpu) = ROF_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "gpu",
)

toc = timeit.default_timer()

Run_time_rof = toc - tic
Qtools = QualityTools(phantom_tm, rof_gpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, rof_gpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_rof = Qtools.ssim(win2d)

print(
    "ROF-TV (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE_rof, ssim_rof[0], Run_time_rof
    )
)
# %%
print("#############FGP TV CPU####################")
# set parameters
pars = {
    "algorithm": FGP_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 930,
    "tolerance_constant": 0.0,
    "methodTV": 0,
    "nonneg": 0,
}

tic = timeit.default_timer()
(fgp_cpu3D, infoFGP) = FGP_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["tolerance_constant"],
    pars["methodTV"],
    pars["nonneg"],
    "cpu",
)
toc = timeit.default_timer()

Run_time_fgp = toc - tic
Qtools = QualityTools(phantom_tm, fgp_cpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, fgp_cpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_fgp = Qtools.ssim(win2d)

print(
    "FGP-TV (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE_rof, ssim_fgp[0], Run_time_fgp
    )
)
# %%
print("#############FGP TV GPU####################")
# set parameters
pars = {
    "algorithm": FGP_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 930,
    "tolerance_constant": 0.0,
    "methodTV": 0,
    "nonneg": 0,
}

tic = timeit.default_timer()
(fgp_gpu3D, infogpu) = FGP_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["tolerance_constant"],
    pars["methodTV"],
    pars["nonneg"],
    "gpu",
)
toc = timeit.default_timer()

Run_time_fgp = toc - tic
Qtools = QualityTools(phantom_tm, fgp_gpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, fgp_gpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_fgp = Qtools.ssim(win2d)

print(
    "FGP-TV (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE_rof, ssim_fgp[0], Run_time_fgp
    )
)
# %%
print("#############SB TV CPU####################")
# set parameters
pars = {
    "algorithm": SB_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 225,
    "tolerance_constant": 0.0,
    "methodTV": 0,
}

tic = timeit.default_timer()
(sb_cpu3D, info_vec_cpu) = SB_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["tolerance_constant"],
    pars["methodTV"],
    "cpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, sb_cpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, sb_cpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "SB-TV (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############SB TV GPU####################")
# set parameters
pars = {
    "algorithm": SB_TV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "number_of_iterations": 225,
    "tolerance_constant": 0.0,
    "methodTV": 0,
}

tic = timeit.default_timer()
(sb_gpu3D, info_vec_gpu) = SB_TV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["number_of_iterations"],
    pars["tolerance_constant"],
    pars["methodTV"],
    "gpu",
)

toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, sb_gpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, sb_gpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "SB-TV (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############NDF CPU####################")
# set parameters
pars = {
    "algorithm": NDF,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "edge_parameter": 0.017,
    "number_of_iterations": 530,
    "time_marching_parameter": 0.01,
    "penalty_type": 1,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(ndf_cpu3D, info_vec_cpu) = NDF(
    pars["input"],
    pars["regularisation_parameter"],
    pars["edge_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["penalty_type"],
    pars["tolerance_constant"],
    "cpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, ndf_cpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, ndf_cpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "NDF (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############NDF GPU####################")
# set parameters
pars = {
    "algorithm": NDF,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "edge_parameter": 0.017,
    "number_of_iterations": 530,
    "time_marching_parameter": 0.01,
    "penalty_type": 1,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(ndf_gpu3D, info_vec_gpu) = NDF(
    pars["input"],
    pars["regularisation_parameter"],
    pars["edge_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["penalty_type"],
    pars["tolerance_constant"],
    "gpu",
)

toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, ndf_gpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, ndf_gpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "NDF (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############Diff4th CPU####################")
# set parameters
pars = {
    "algorithm": Diff4th,
    "input": phantom_noise,
    "regularisation_parameter": 4.5,
    "edge_parameter": 0.035,
    "number_of_iterations": 2425,
    "time_marching_parameter": 0.001,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(diff4th_cpu3D, info_vec_cpu) = Diff4th(
    pars["input"],
    pars["regularisation_parameter"],
    pars["edge_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "cpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, diff4th_cpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(
    phantom_tm[sliceSel, :, :] * 255, diff4th_cpu3D[sliceSel, :, :] * 235
)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "Diff4th (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############Diff4th GPU####################")
# set parameters
pars = {
    "algorithm": Diff4th,
    "input": phantom_noise,
    "regularisation_parameter": 4.5,
    "edge_parameter": 0.035,
    "number_of_iterations": 2425,
    "time_marching_parameter": 0.001,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(diff4th_gpu3D, info_vec_gpu) = Diff4th(
    pars["input"],
    pars["regularisation_parameter"],
    pars["edge_parameter"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "gpu",
)

toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, diff4th_gpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(
    phantom_tm[sliceSel, :, :] * 255, diff4th_gpu3D[sliceSel, :, :] * 235
)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "Diff4th (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############TGV CPU####################")
# set parameters
pars = {
    "algorithm": TGV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "alpha1": 1.0,
    "alpha0": 2.0,
    "number_of_iterations": 1000,
    "LipshitzConstant": 12,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(tgv_cpu3D, info_vec_cpu) = TGV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["alpha1"],
    pars["alpha0"],
    pars["number_of_iterations"],
    pars["LipshitzConstant"],
    pars["tolerance_constant"],
    "cpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, tgv_cpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, tgv_cpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "TGV (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############TGV GPU####################")
# set parameters
pars = {
    "algorithm": TGV,
    "input": phantom_noise,
    "regularisation_parameter": 0.06,
    "alpha1": 1.0,
    "alpha0": 2.0,
    "number_of_iterations": 7845,
    "LipshitzConstant": 12,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(tgv_gpu3D, info_vec_gpu) = TGV(
    pars["input"],
    pars["regularisation_parameter"],
    pars["alpha1"],
    pars["alpha0"],
    pars["number_of_iterations"],
    pars["LipshitzConstant"],
    pars["tolerance_constant"],
    "gpu",
)

toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, tgv_gpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[sliceSel, :, :] * 255, tgv_gpu3D[sliceSel, :, :] * 235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "TGV (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############ROF-LLT CPU####################")
# set parameters
pars = {
    "algorithm": LLT_ROF,
    "input": phantom_noise,
    "regularisation_parameterROF": 0.03,
    "regularisation_parameterLLT": 0.015,
    "number_of_iterations": 1000,
    "time_marching_parameter": 0.00025,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(rofllt_cpu3D, info_vec_cpu) = LLT_ROF(
    pars["input"],
    pars["regularisation_parameterROF"],
    pars["regularisation_parameterLLT"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "cpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, rofllt_cpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(
    phantom_tm[sliceSel, :, :] * 255, rofllt_cpu3D[sliceSel, :, :] * 235
)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "ROF-LLT  (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)
# %%
print("#############ROF-LLT  GPU####################")
# set parameters
pars = {
    "algorithm": LLT_ROF,
    "input": phantom_noise,
    "regularisation_parameterROF": 0.03,
    "regularisation_parameterLLT": 0.015,
    "number_of_iterations": 8000,
    "time_marching_parameter": 0.00025,
    "tolerance_constant": 0.0,
}

tic = timeit.default_timer()
(rofllt_gpu3D, info_vec_gpu) = LLT_ROF(
    pars["input"],
    pars["regularisation_parameterROF"],
    pars["regularisation_parameterLLT"],
    pars["number_of_iterations"],
    pars["time_marching_parameter"],
    pars["tolerance_constant"],
    "gpu",
)
toc = timeit.default_timer()

Run_time = toc - tic
Qtools = QualityTools(phantom_tm, rofllt_gpu3D)
RMSE = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(
    phantom_tm[sliceSel, :, :] * 255, rofllt_gpu3D[sliceSel, :, :] * 235
)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim = Qtools.ssim(win2d)

print(
    "ROF-LLT  (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(
        RMSE, ssim[0], Run_time
    )
)

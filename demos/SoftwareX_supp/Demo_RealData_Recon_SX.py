#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This demo scripts support the following publication:
"CCPi-Regularisation Toolkit for computed tomographic image reconstruction with
proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Martin J. Turner,
 Philip J. Withers; Software X, 2019
____________________________________________________________________________
* Reads real tomographic data (stored at Zenodo)
--- https://doi.org/10.5281/zenodo.2578893
* Reconstructs using ToMoBAR software
* Saves reconstructed images
____________________________________________________________________________
>>>>> Dependencies: <<<<<
1. ASTRA toolbox: conda install -c astra-toolbox astra-toolbox
2. tomobar: conda install -c dkazanc tomobar
or install from https://github.com/dkazanc/ToMoBAR
3. libtiff if one needs to save tiff images:
    install pip install libtiff

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
GPLv3 license (ASTRA toolbox)
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tomobar.supp.suppTools import normaliser
import time

# load dendritic projection data
h5f = h5py.File("data/DendrData_3D.h5", "r")
dataRaw = h5f["dataRaw"][:]
flats = h5f["flats"][:]
darks = h5f["darks"][:]
angles_rad = h5f["angles_rad"][:]
h5f.close()
# %%
# normalise the data [detectorsVert, Projections, detectorsHoriz]
data_norm = normaliser(dataRaw, flats, darks, log="log")
del dataRaw, darks, flats

intens_max = 2.3
plt.figure()
plt.subplot(131)
plt.imshow(data_norm[:, 150, :], vmin=0, vmax=intens_max)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(data_norm[300, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(data_norm[:, :, 600], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()

detectorHoriz = np.size(data_norm, 2)
det_y_crop = [i for i in range(0, detectorHoriz - 22)]
N_size = 950  # reconstruction domain
time_label = int(time.time())
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=np.size(
        det_y_crop
    ),  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimV=100,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device="gpu",
)

FBPrec = RectoolsDIR.FBP(data_norm[0:100, :, det_y_crop])

sliceSel = 50
max_val = 0.003
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec[sliceSel, :, :], vmin=0, vmax=max_val, cmap="gray")
plt.title("FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec[:, sliceSel, :], vmin=0, vmax=max_val, cmap="gray")
plt.title("FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec[:, :, sliceSel], vmin=0, vmax=max_val, cmap="gray")
plt.title("FBP Reconstruction, sagittal view")
plt.show()

# saving to tiffs (16bit)
"""
from libtiff import TIFF
FBPrec += np.abs(np.min(FBPrec))
multiplier = (int)(65535/(np.max(FBPrec)))

# saving to tiffs (16bit)
for i in range(0,np.size(FBPrec,0)):
    tiff = TIFF.open('Dendr_FBP'+'_'+str(i)+'.tiff', mode='w')
    tiff.write_image(np.uint16(FBPrec[i,:,:]*multiplier))
    tiff.close()
"""
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM method using tomobar software")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class ONCE
from tomobar.methodsIR import RecToolsIR

RectoolsIR = RecToolsIR(
    DetectorsDimH=np.size(
        det_y_crop
    ),  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimV=100,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    datafidelity="LS",  # data fidelity, choose LS, PWLS, GH (wip), Students t (wip)
    nonnegativity="ENABLE",  # enable nonnegativity constraint (set to 'ENABLE')
    OS_number=None,  # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
    tolerance=0.0,  # tolerance to stop inner (regularisation) iterations earlier
    device="gpu",
)
# %%
print("Reconstructing with ADMM method using SB-TV penalty")
RecADMM_reg_sbtv = RectoolsIR.ADMM(
    data_norm[0:100, :, det_y_crop],
    rho_const=2000.0,
    iterationsADMM=15,
    regularisation="SB_TV",
    regularisation_parameter=0.00085,
    regularisation_iterations=50,
)

sliceSel = 50
max_val = 0.003
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_reg_sbtv[sliceSel, :, :], vmin=0, vmax=max_val, cmap="gray")
plt.title("3D ADMM-SB-TV Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_reg_sbtv[:, sliceSel, :], vmin=0, vmax=max_val, cmap="gray")
plt.title("3D ADMM-SB-TV Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_reg_sbtv[:, :, sliceSel], vmin=0, vmax=max_val, cmap="gray")
plt.title("3D ADMM-SB-TV Reconstruction, sagittal view")
plt.show()


# saving to tiffs (16bit)
"""
from libtiff import TIFF
multiplier = (int)(65535/(np.max(RecADMM_reg_sbtv)))
for i in range(0,np.size(RecADMM_reg_sbtv,0)):
    tiff = TIFF.open('Dendr_ADMM_SBTV'+'_'+str(i)+'.tiff', mode='w')
    tiff.write_image(np.uint16(RecADMM_reg_sbtv[i,:,:]*multiplier))
    tiff.close()
"""
# Saving recpnstructed data with a unique time label
np.save("Dendr_ADMM_SBTV" + str(time_label) + ".npy", RecADMM_reg_sbtv)
del RecADMM_reg_sbtv
# %%
print("Reconstructing with ADMM method using ROF-LLT penalty")
RecADMM_reg_rofllt = RectoolsIR.ADMM(
    data_norm[0:100, :, det_y_crop],
    rho_const=2000.0,
    iterationsADMM=15,
    regularisation="LLT_ROF",
    regularisation_parameter=0.0009,
    regularisation_parameter2=0.0007,
    time_marching_parameter=0.001,
    regularisation_iterations=550,
)

sliceSel = 50
max_val = 0.003
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_reg_rofllt[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-ROFLLT Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_reg_rofllt[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-ROFLLT Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_reg_rofllt[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D ADMM-ROFLLT Reconstruction, sagittal view")
plt.show()

# saving to tiffs (16bit)
"""
from libtiff import TIFF
multiplier = (int)(65535/(np.max(RecADMM_reg_rofllt)))
for i in range(0,np.size(RecADMM_reg_rofllt,0)):
    tiff = TIFF.open('Dendr_ADMM_ROFLLT'+'_'+str(i)+'.tiff', mode='w')
    tiff.write_image(np.uint16(RecADMM_reg_rofllt[i,:,:]*multiplier))
    tiff.close()
"""

# Saving recpnstructed data with a unique time label
np.save("Dendr_ADMM_ROFLLT" + str(time_label) + ".npy", RecADMM_reg_rofllt)
del RecADMM_reg_rofllt
# %%
print("Reconstructing with ADMM method using TGV penalty")
RecADMM_reg_tgv = RectoolsIR.ADMM(
    data_norm[0:100, :, det_y_crop],
    rho_const=2000.0,
    iterationsADMM=15,
    regularisation="TGV",
    regularisation_parameter=0.01,
    regularisation_iterations=500,
)

sliceSel = 50
max_val = 0.003
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_reg_tgv[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-TGV Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_reg_tgv[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-TGV Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_reg_tgv[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D ADMM-TGV Reconstruction, sagittal view")
plt.show()

# saving to tiffs (16bit)
"""
from libtiff import TIFF
multiplier = (int)(65535/(np.max(RecADMM_reg_tgv)))
for i in range(0,np.size(RecADMM_reg_tgv,0)):
    tiff = TIFF.open('Dendr_ADMM_TGV'+'_'+str(i)+'.tiff', mode='w')
    tiff.write_image(np.uint16(RecADMM_reg_tgv[i,:,:]*multiplier))
    tiff.close()
"""
# Saving recpnstructed data with a unique time label
np.save("Dendr_ADMM_TGV" + str(time_label) + ".npy", RecADMM_reg_tgv)
del RecADMM_reg_tgv
# %%

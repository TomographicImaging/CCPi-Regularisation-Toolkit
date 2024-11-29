#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This demo scripts support the following publication: 
"CCPi-Regularisation Toolkit for computed tomographic image reconstruction with 
proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Martin J. Turner,
 Philip J. Withers; Software X, 2019
____________________________________________________________________________
* Runs TomoPhantom software to simulate tomographic projection data with
some imaging errors and noise
* Saves the data into hdf file to be uploaded in reconstruction scripts
__________________________________________________________________________

>>>>> Dependencies: <<<<<
1. TomoPhantom software for phantom and data generation

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
Apache 2.0 license
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.flatsgen import flats
from tomophantom.supp.normraw import normaliser_sim

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 16  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel, :, :], vmin=0, vmax=1)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_tm[:, sliceSel, :], vmin=0, vmax=1)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_tm[:, :, sliceSel], vmin=0, vmax=1)
plt.title("3D Phantom, sagittal view")
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.35 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)
# %%
print("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)

intens_max = N_size
sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt[:, sliceSel, :], vmin=0, vmax=intens_max)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_analyt[:, :, sliceSel], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()
# %%
print("Simulate flat fields, add noise and normalise projections...")
flatsnum = 20  # generate 20 flat fields
flatsSIM = flats(
    Vert_det,
    Horiz_det,
    maxheight=0.1,
    maxthickness=3,
    sigma_noise=0.2,
    sigmasmooth=3,
    flatsnum=flatsnum,
)

plt.figure()
plt.imshow(flatsSIM[0, :, :], vmin=0, vmax=1)
plt.title("A selected simulated flat-field")
# %%
# Apply normalisation of data and add noise
flux_intensity = 60000  # controls the level of noise
sigma_flats = (
    0.01  # contro the level of noise in flats (higher creates more ring artifacts)
)
projData3D_norm = normaliser_sim(
    projData3D_analyt, flatsSIM, sigma_flats, flux_intensity
)

intens_max = N_size
sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_norm[:, sliceSel, :], vmin=0, vmax=intens_max)
plt.title("2D Projection (erroneous)")
plt.subplot(132)
plt.imshow(projData3D_norm[sliceSel, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_norm[:, :, sliceSel], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()
# %%
import h5py
import time

time_label = int(time.time())
# Saving generated data with a unique time label
h5f = h5py.File("TomoSim_data" + str(time_label) + ".h5", "w")
h5f.create_dataset("phantom", data=phantom_tm)
h5f.create_dataset("projdata_norm", data=projData3D_norm)
h5f.create_dataset("proj_angles", data=angles_rad)
h5f.close()
# %%

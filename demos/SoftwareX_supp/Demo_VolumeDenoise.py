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
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, LLT_ROF, NDF, Diff4th
#%%
print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 9 # select a model number from the library
N_size = 256 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc=timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

# adding normally distributed noise
artifacts_add = ArtifactsClass(phantom_tm)
phantom_noise = artifacts_add.noise(sigma=0.1,noisetype='Gaussian')

sliceSel = int(0.5*N_size)
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(phantom_noise[sliceSel,:,:],vmin=0, vmax=1.4)
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(phantom_noise[:,sliceSel,:],vmin=0, vmax=1.4)
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(phantom_noise[:,:,sliceSel],vmin=0, vmax=1.4)
plt.title('3D Phantom, sagittal view')
plt.show()
#%%
print ("____________________Applying regularisers_______________________")

print ("#############ROF TV CPU####################")
# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : phantom_noise,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 100,\
        'time_marching_parameter': 0.0025
        }

tic=timeit.default_timer()
rof_cpu3D = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],'cpu')
toc=timeit.default_timer()

Run_time_rof = toc - tic
Qtools = QualityTools(phantom_tm, rof_cpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[128,:,:]*255, rof_cpu3D[128,:,:]*235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_rof = Qtools.ssim(win2d)

print("ROF-TV (cpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(RMSE_rof,ssim_rof[0],Run_time_rof))
#%%
print ("#############ROF TV GPU####################")
# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : phantom_noise,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 600,\
        'time_marching_parameter': 0.0025
        }

tic=timeit.default_timer()
rof_gpu3D = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],'gpu')
toc=timeit.default_timer()

Run_time_rof = toc - tic
Qtools = QualityTools(phantom_tm, rof_gpu3D)
RMSE_rof = Qtools.rmse()

# SSIM measure
Qtools = QualityTools(phantom_tm[128,:,:]*255, rof_gpu3D[128,:,:]*235)
win = np.array([gaussian(11, 1.5)])
win2d = win * (win.T)
ssim_rof = Qtools.ssim(win2d)

print("ROF-TV (gpu) ____ RMSE: {}, MMSIM: {}, run time: {} sec".format(RMSE_rof,ssim_rof[0],Run_time_rof))

#%%


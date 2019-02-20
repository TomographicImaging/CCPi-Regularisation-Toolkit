#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This demo scripts support the following publication: 
"CCPi-Regularisation Toolkit for computed tomographic image reconstruction with 
proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Mark Basham, 
Martin J. Turner, Philip J. Withers and Alun Ashton; Software X, 2019
____________________________________________________________________________
* Runs TomoPhantom software to simulate tomographic projection data with
some imaging errors and noise
* Saves the data into hdf file to be uploaded in reconstruction scripts
____________________________________________________________________________

>>>>> Dependencies: <<<<<
1. TomoPhantom software for data generation

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
Apache 2.0 license
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tomorec.supp.suppTools import normaliser

# load dendritic data
datadict = scipy.io.loadmat('Rawdata_1_frame3D.mat')
# extract data (print(datadict.keys()))
dataRaw = datadict['Rawdata3D']
angles = datadict['AnglesAr']
flats = datadict['flats3D']
darks=  datadict['darks3D']

dataRaw = np.swapaxes(dataRaw,0,1)
#%%
#flats2 = np.zeros((np.size(flats,0),1, np.size(flats,1)), dtype='float32')
#flats2[:,0,:] = flats[:]
#darks2 = np.zeros((np.size(darks,0),1, np.size(darks,1)), dtype='float32')
#darks2[:,0,:] = darks[:]

# normalise the data, required format is [detectorsHoriz, Projections, Slices]
data_norm = normaliser(dataRaw, flats, darks, log='log')

#dataRaw = np.float32(np.divide(dataRaw, np.max(dataRaw).astype(float)))

intens_max = 70
plt.figure() 
plt.subplot(131)
plt.imshow(data_norm[:,150,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(data_norm[:,:,300],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(data_norm[600,:,:],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()


detectorHoriz = np.size(data_norm,0)
N_size = 1000
slice_to_recon = 0 # select which slice to reconstruct
angles_rad = angles*(np.pi/180.0)
#%%

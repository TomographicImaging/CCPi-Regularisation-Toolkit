#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This demo scripts support the following publication: 
"CCPi-Regularisation Toolkit for computed tomographic image reconstruction with 
proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Mark Basham, 
Martin J. Turner, Philip J. Withers and Alun Ashton; Software X, 2019
____________________________________________________________________________
* Reads real tomographic data (stored at Zenodo)
* Reconstructs using TomoRec software
* Saves reconstructed images 
____________________________________________________________________________
>>>>> Dependencies: <<<<<
1. ASTRA toolbox: conda install -c astra-toolbox astra-toolbox
2. TomoRec: conda install -c dkazanc tomorec
or install from https://github.com/dkazanc/TomoRec
3. TomoPhantom software for data generation

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
GPLv3 license (ASTRA toolbox)
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
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions                   
                    device='gpu')

FBPrec = RectoolsDIR.FBP(np.transpose(data_norm[:,:,slice_to_recon]))

plt.figure()
plt.imshow(FBPrec[150:550,150:550], vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')

from tomorec.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod(np.transpose(dataRaw[:,:,slice_to_recon])) # calculate Lipschitz constant (run once to initilise)

RecFISTA_os_pwls = Rectools.FISTA(np.transpose(data_norm[:,:,slice_to_recon]), \
                             np.transpose(dataRaw[:,:,slice_to_recon]), \
                             iterationsFISTA = 15, \
                             lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_os_pwls[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.imshow(RecFISTA_os_pwls, vmin=0, vmax=0.004, cmap="gray")
plt.title('FISTA PWLS-OS reconstruction')
plt.show()
#fig.savefig('dendr_PWLS.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_TV = Rectools.FISTA(np.transpose(data_norm[:,:,slice_to_recon]), \
                              np.transpose(dataRaw[:,:,slice_to_recon]), \
                              iterationsFISTA = 15, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 200,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_TV.png', dpi=200)
#%%
"""
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-Diff4th method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_Diff4th = Rectools.FISTA(np.transpose(data_norm[:,:,slice_to_recon]), \
                              np.transpose(dataRaw[:,:,slice_to_recon]), \
                              iterationsFISTA = 15, \
                              regularisation = 'DIFF4th', \
                              regularisation_parameter = 0.1,\
                              time_marching_parameter = 0.001,\
                              edge_param = 0.003,\
                              regularisation_iterations = 600,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_Diff4th[150:550,150:550], vmin=0, vmax=0.004, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-Diff4th reconstruction')
plt.show()
#fig.savefig('dendr_Diff4th.png', dpi=200)
"""
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-ROF_LLT method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_rofllt = Rectools.FISTA(np.transpose(data_norm[:,:,slice_to_recon]), \
                              np.transpose(dataRaw[:,:,slice_to_recon]), \
                              iterationsFISTA = 15, \
                              regularisation = 'LLT_ROF', \
                              regularisation_parameter = 0.000007,\
                              regularisation_parameter2 = 0.0004,\
                              regularisation_iterations = 350,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_rofllt[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-ROF-LLT reconstruction')
plt.show()
#fig.savefig('dendr_ROFLLT.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TGV method %%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_tgv = Rectools.FISTA(np.transpose(data_norm[:,:,slice_to_recon]), \
                              np.transpose(dataRaw[:,:,slice_to_recon]), \
                              iterationsFISTA = 15, \
                              regularisation = 'TGV', \
                              regularisation_parameter = 0.001,\
                              TGV_alpha2 = 0.001,\
                              TGV_alpha1 = 2.0,\
                              TGV_LipschitzConstant = 24,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_tgv[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TGV reconstruction')
plt.show()
#fig.savefig('dendr_TGV.png', dpi=200)

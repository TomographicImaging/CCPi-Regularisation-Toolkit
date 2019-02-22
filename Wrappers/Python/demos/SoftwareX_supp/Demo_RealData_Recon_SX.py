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

@author: Daniil Kazantsev, e:mail daniil.kazantsev@diamond.ac.uk
GPLv3 license (ASTRA toolbox)
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tomorec.supp.suppTools import normaliser


# load dendritic projection data
h5f = h5py.File('data/DendrData_3D.h5','r')
dataRaw = h5f['dataRaw'][:]
flats = h5f['flats'][:]
darks = h5f['darks'][:]
angles_rad = h5f['angles_rad'][:]
h5f.close()
#%%
# normalise the data, required format is [detectorsHoriz, Projections, Slices]
data_norm = normaliser(dataRaw, flats, darks, log='log')
del dataRaw, darks, flats

intens_max = 2
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
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsDIR import RecToolsDIR
det_y_crop = [i for i in range(0,detectorHoriz-22)]

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions                   
                    device='gpu')

FBPrec = RectoolsDIR.FBP(np.transpose(data_norm[det_y_crop,:,slice_to_recon]))

plt.figure()
plt.imshow(FBPrec[150:550,150:550], vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')

#%%
from tomorec.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)

RecFISTA_os_pwls = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_to_recon]), \
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
RecFISTA_pwls_os_TV = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_to_recon]), \
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

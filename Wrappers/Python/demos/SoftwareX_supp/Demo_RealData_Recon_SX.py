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
# normalise the data [detectorsVert, Projections, detectorsHoriz]
data_norm = normaliser(dataRaw, flats, darks, log='log')
del dataRaw, darks, flats

intens_max = 2
plt.figure() 
plt.subplot(131)
plt.imshow(data_norm[:,150,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(data_norm[300,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(data_norm[:,:,600],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()


detectorHoriz = np.size(data_norm,2)
det_y_crop = [i for i in range(0,detectorHoriz-22)]
N_size = 950 # reconstruction domain
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = 10,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBPrec = RectoolsDIR.FBP(data_norm[0:10,:,det_y_crop])

plt.figure()
#plt.imshow(FBPrec[0,150:550,150:550], vmin=0, vmax=0.005, cmap="gray")
plt.imshow(FBPrec[0,:,:], vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with ADMM method using TomoRec software")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise TomoRec ITERATIVE reconstruction class ONCE
from tomorec.methodsIR import RecToolsIR
RectoolsIR = RecToolsIR(DetectorsDimH =  np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = 5,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')
#%%
print ("Reconstructing with ADMM method using ROF-TV penalty")

RecADMM_reg_roftv = RectoolsIR.ADMM(data_norm[0:5,:,det_y_crop],
                              rho_const = 2000.0, \
                              iterationsADMM = 3, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.001,\
                              regularisation_iterations = 80)


sliceSel = 2
max_val = 0.005
plt.figure() 
plt.subplot(131)
plt.imshow(RecADMM_reg_roftv[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D ADMM-ROF-TV Reconstruction, axial view')

plt.subplot(132)
plt.imshow(RecADMM_reg_roftv[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D ADMM-ROF-TV Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(RecADMM_reg_roftv[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D ADMM-ROF-TV Reconstruction, sagittal view')
plt.show()
#%%
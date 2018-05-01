#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of CPU inpainters
@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from scipy import io
from ccpi.filters.regularisers import NDF_INP, NVM_INP
from qualitymetrics import rmse
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'maskData':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################

# read sinogram and the mask
filename = os.path.join(".." , ".." , ".." , "data" ,"SinoInpaint.mat")
sino = io.loadmat(filename)
sino_full = sino.get('Sinogram')
Mask = sino.get('Mask')
[angles_dim,detectors_dim] = sino_full.shape
sino_full = sino_full/np.max(sino_full)
#apply mask to sinogram
sino_cut = sino_full*(1-Mask)
sino_cut_new = np.zeros((angles_dim,detectors_dim),'float32')
#sino_cut_new = sino_cut.copy(order='c')
sino_cut_new[:] = sino_cut[:]
mask = np.zeros((angles_dim,detectors_dim),'uint8')
#mask =Mask.copy(order='c')
mask[:] = Mask[:]

plt.figure(1)
plt.subplot(121)
plt.imshow(sino_cut_new,vmin=0.0, vmax=1)
plt.title('Missing Data sinogram')
plt.subplot(122)
plt.imshow(mask)
plt.title('Mask')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting using linear diffusion (2D)__")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(2)
plt.suptitle('Performance of linear inpainting using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sino_cut_new,cmap="gray")

# set parameters
pars = {'algorithm' : NDF_INP, \
        'input' : sino_cut_new,\
        'maskData' : mask,\
        'regularisation_parameter':1000,\
        'edge_parameter':0.0,\
        'number_of_iterations' :1000 ,\
        'time_marching_parameter':0.000075,\
        'penalty_type':1
        }
        
start_time = timeit.default_timer()
ndf_inp_linear = NDF_INP(pars['input'],
              pars['maskData'],
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'])
             
rms = rmse(sino_full, ndf_inp_linear)
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_inp_linear, cmap="gray")
plt.title('{}'.format('Linear diffusion inpainting results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_Inpainting using nonlinear diffusion (2D)_")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(3)
plt.suptitle('Performance of nonlinear diffusion inpainting using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sino_cut,cmap="gray")

# set parameters
pars = {'algorithm' : NDF_INP, \
        'input' : sino_cut_new,\
        'maskData' : mask,\
        'regularisation_parameter':80,\
        'edge_parameter':0.00009,\
        'number_of_iterations' :1500 ,\
        'time_marching_parameter':0.000008,\
        'penalty_type':1
        }
        
start_time = timeit.default_timer()
ndf_inp_nonlinear = NDF_INP(pars['input'],
              pars['maskData'],
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'])
             
rms = rmse(sino_full, ndf_inp_nonlinear)
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_inp_nonlinear, cmap="gray")
plt.title('{}'.format('Nonlinear diffusion inpainting results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Inpainting using nonlocal vertical marching")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(4)
plt.suptitle('Performance of NVM inpainting using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sino_cut,cmap="gray")

# set parameters
pars = {'algorithm' : NVM_INP, \
        'input' : sino_cut_new,\
        'maskData' : mask,\
        'SW_increment': 1,\
        'number_of_iterations' :20
        }
        
start_time = timeit.default_timer()
nvm_inp = NVM_INP(pars['input'],
              pars['maskData'],
              pars['SW_increment'],
              pars['number_of_iterations'])
             
rms = rmse(sino_full, nvm_inp)
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(nvm_inp, cmap="gray")
plt.title('{}'.format('Nonlocal Vertical Marching inpainting results'))
#%%

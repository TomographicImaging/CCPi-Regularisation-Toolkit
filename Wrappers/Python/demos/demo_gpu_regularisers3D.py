#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of GPU regularisers

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, LLT_ROF, FGP_dTV, NDF, Diff4th
from qualitymetrics import rmse
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'refdata':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################
#%%
filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")

# read image
Im = plt.imread(filename)                     
Im = np.asarray(Im, dtype='float32')

Im = Im/255
perc = 0.05
u0 = Im + np.random.normal(loc = 0 ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
u_ref = Im + np.random.normal(loc = 0 ,
                                  scale = 0.01 * Im , 
                                  size = np.shape(Im))
(N,M) = np.shape(u0)
# map the u0 u0->u0>0
# f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')
"""
M = M-100
u_ref2 = np.zeros([N,M],dtype='float32')
u_ref2[:,0:M] = u_ref[:,0:M]
u_ref = u_ref2
del u_ref2

u02 = np.zeros([N,M],dtype='float32')
u02[:,0:M] = u0[:,0:M]
u0 = u02
del u02

Im2 = np.zeros([N,M],dtype='float32')
Im2[:,0:M] = Im[:,0:M]
Im = Im2
del Im2
"""

#%%
slices = 20

filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")
Im = plt.imread(filename)
Im = np.asarray(Im, dtype='float32')

Im = Im/255
perc = 0.05

noisyVol = np.zeros((slices,N,N),dtype='float32')
noisyRef = np.zeros((slices,N,N),dtype='float32')
idealVol = np.zeros((slices,N,N),dtype='float32')

for i in range (slices):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    noisyRef[i,:,:] = Im + np.random.normal(loc = 0 , scale = 0.01 * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________ROF-TV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of ROF-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy 15th slice of a volume')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 500,\
        'time_marching_parameter': 0.0025        
        }
print ("#############ROF TV GPU####################")
start_time = timeit.default_timer()
rof_gpu3D = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],'gpu')
rms = rmse(idealVol, rof_gpu3D)
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
imgplot = plt.imshow(rof_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using ROF-TV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-TV (3D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :300 ,\
        'tolerance_constant':0.00001,\
        'methodTV': 0 ,\
        'nonneg': 0 ,\
        'printingOut': 0 
        }

print ("#############FGP TV GPU####################")
start_time = timeit.default_timer()
fgp_gpu3D = FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'gpu')

rms = rmse(idealVol, fgp_gpu3D)
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
imgplot = plt.imshow(fgp_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using FGP-TV'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________SB-TV (3D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of SB-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :100 ,\
        'tolerance_constant':1e-05,\
        'methodTV': 0 ,\
        'printingOut': 0 
        }

print ("#############SB TV GPU####################")
start_time = timeit.default_timer()
sb_gpu3D = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['printingOut'],'gpu')

rms = rmse(idealVol, sb_gpu3D)
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
imgplot = plt.imshow(sb_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using SB-TV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________LLT-ROF (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of LLT-ROF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : LLT_ROF, \
        'input' : noisyVol,\
        'regularisation_parameterROF':0.04, \
        'regularisation_parameterLLT':0.015, \
        'number_of_iterations' :300 ,\
        'time_marching_parameter' :0.0025 ,\
        }

print ("#############LLT ROF CPU####################")
start_time = timeit.default_timer()
lltrof_gpu3D = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],'gpu')

rms = rmse(idealVol, lltrof_gpu3D)
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
imgplot = plt.imshow(lltrof_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using LLT-ROF'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________NDF-TV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of NDF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : NDF, \
        'input' : noisyVol,\
        'regularisation_parameter':0.025, \
        'edge_parameter':0.015,\
        'number_of_iterations' :500 ,\
        'time_marching_parameter':0.025,\
        'penalty_type':  1
        }

print ("#############NDF GPU####################")
start_time = timeit.default_timer()
ndf_gpu3D = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],'gpu')

rms = rmse(idealVol, ndf_gpu3D)
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
imgplot = plt.imshow(ndf_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using NDF'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (3D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of DIFF4th regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : Diff4th, \
        'input' : noisyVol,\
        'regularisation_parameter':3.5, \
        'edge_parameter':0.02,\
        'number_of_iterations' :300 ,\
        'time_marching_parameter':0.0015
        }
        
print ("#############DIFF4th CPU################")
start_time = timeit.default_timer()
diff4_gpu3D = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],'gpu')
             
rms = rmse(idealVol, diff4_gpu3D)
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
imgplot = plt.imshow(diff4_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-dTV (3D)________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-dTV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : FGP_dTV, \
        'input' : noisyVol,\
        'refdata' : noisyRef,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :300 ,\
        'tolerance_constant':0.00001,\
        'eta_const':0.2,\
        'methodTV': 0 ,\
        'nonneg': 0 ,\
        'printingOut': 0 
        }

print ("#############FGP TV GPU####################")
start_time = timeit.default_timer()
fgp_dTV_gpu3D = FGP_dTV(pars['input'],
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'],
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'gpu')

rms = rmse(idealVol, fgp_dTV_gpu3D)
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
imgplot = plt.imshow(fgp_dTV_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using FGP-dTV'))
#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of CPU regularisers 

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, FGP_dTV, TNV, NDF, DIFF4th
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

# change dims to check that modules work with non-squared images
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
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________ROF-TV (2D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(1)
plt.suptitle('Performance of ROF-TV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : u0,\
        'regularisation_parameter':0.04,\
        'number_of_iterations': 1200,\
        'time_marching_parameter': 0.0025        
        }
print ("#############ROF TV CPU####################")
start_time = timeit.default_timer()
rof_cpu = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],'cpu')
rms = rmse(Im, rof_cpu)
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
imgplot = plt.imshow(rof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(2)
plt.suptitle('Performance of FGP-TV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : u0,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :2000 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'nonneg': 0 ,\
        'printingOut': 0 
        }
        
print ("#############FGP TV CPU####################")
start_time = timeit.default_timer()
fgp_cpu = FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'cpu')  
             
             
rms = rmse(Im, fgp_cpu)
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
imgplot = plt.imshow(fgp_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________SB-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(3)
plt.suptitle('Performance of SB-TV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : u0,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :150 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'printingOut': 0 
        }
        
print ("#############SB TV CPU####################")
start_time = timeit.default_timer()
sb_cpu = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['printingOut'],'cpu')  
             
             
rms = rmse(Im, sb_cpu)
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
imgplot = plt.imshow(sb_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("________________NDF (2D)___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(4)
plt.suptitle('Performance of NDF regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : NDF, \
        'input' : u0,\
        'regularisation_parameter':0.025, \
        'edge_parameter':0.015,\
        'number_of_iterations' :500 ,\
        'time_marching_parameter':0.025,\
        'penalty_type':1
        }
        
print ("#############NDF CPU################")
start_time = timeit.default_timer()
ndf_cpu = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],'cpu')  
             
rms = rmse(Im, ndf_cpu)
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
imgplot = plt.imshow(ndf_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (2D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(5)
plt.suptitle('Performance of DIFF4th regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : DIFF4th, \
        'input' : u0,\
        'regularisation_parameter':3.5, \
        'edge_parameter':0.02,\
        'number_of_iterations' :500 ,\
        'time_marching_parameter':0.0015
        }
        
print ("#############DIFF4th CPU################")
start_time = timeit.default_timer()
diff4_cpu = DIFF4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],'cpu')
             
rms = rmse(Im, diff4_cpu)
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
imgplot = plt.imshow(diff4_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_____________FGP-dTV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(6)
plt.suptitle('Performance of FGP-dTV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_dTV, \
        'input' : u0,\
        'refdata' : u_ref,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :2000 ,\
        'tolerance_constant':1e-06,\
        'eta_const':0.2,\
        'methodTV': 0 ,\
        'nonneg': 0 ,\
        'printingOut': 0 
        }
        
print ("#############FGP dTV CPU####################")
start_time = timeit.default_timer()
fgp_dtv_cpu = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'cpu')
             
rms = rmse(Im, fgp_dtv_cpu)
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
imgplot = plt.imshow(fgp_dtv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("__________Total nuclear Variation__________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(7)
plt.suptitle('Performance of TNV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

channelsNo = 5
noisyVol = np.zeros((channelsNo,N,M),dtype='float32')
idealVol = np.zeros((channelsNo,N,M),dtype='float32')

for i in range (channelsNo):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im

# set parameters
pars = {'algorithm' : TNV, \
        'input' : noisyVol,\
        'regularisation_parameter': 0.04, \
        'number_of_iterations' : 200 ,\
        'tolerance_constant':1e-05
        }
        
print ("#############TNV CPU#################")
start_time = timeit.default_timer()
tnv_cpu = TNV(pars['input'],           
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'])
             
rms = rmse(idealVol, tnv_cpu)
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
imgplot = plt.imshow(tnv_cpu[3,:,:], cmap="gray")
plt.title('{}'.format('CPU results'))

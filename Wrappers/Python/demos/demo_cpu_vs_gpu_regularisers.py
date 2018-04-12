#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of CPU implementation against the GPU one

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, FGP_dTV
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

# map the u0 u0->u0>0
# f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________ROF-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(1)
plt.suptitle('Comparison of ROF-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
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
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("##############ROF TV GPU##################")
start_time = timeit.default_timer()
rof_gpu = ROF_TV(pars['input'], 
                     pars['regularisation_parameter'],
                     pars['number_of_iterations'], 
                     pars['time_marching_parameter'],'gpu')
                     
rms = rmse(Im, rof_gpu)
pars['rmse'] = rms
pars['algorithm'] = ROF_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(rof_cpu))
diff_im = abs(rof_cpu - rof_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-TV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(2)
plt.suptitle('Comparison of FGP-TV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : u0,\
        'regularisation_parameter':0.04, \
        'number_of_iterations' :1200 ,\
        'tolerance_constant':0.00001,\
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
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))


print ("##############FGP TV GPU##################")
start_time = timeit.default_timer()
fgp_gpu = FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'gpu')
                                   
rms = rmse(Im, fgp_gpu)
pars['rmse'] = rms
pars['algorithm'] = FGP_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(rof_cpu))
diff_im = abs(fgp_cpu - fgp_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")


print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-dTV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(3)
plt.suptitle('Comparison of FGP-dTV regulariser using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
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
a=fig.add_subplot(1,4,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_dtv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

print ("##############FGP dTV GPU##################")
start_time = timeit.default_timer()
fgp_dtv_gpu = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'],'gpu')
rms = rmse(Im, fgp_dtv_gpu)
pars['rmse'] = rms
pars['algorithm'] = FGP_dTV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_dtv_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))


print ("--------Compare the results--------")
tolerance = 1e-05
diff_im = np.zeros(np.shape(rof_cpu))
diff_im = abs(fgp_dtv_cpu - fgp_dtv_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")
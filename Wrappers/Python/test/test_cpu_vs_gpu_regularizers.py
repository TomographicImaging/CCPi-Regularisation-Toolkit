#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Testing CPU implementation against GPU one

@author: algol
"""

import matplotlib.pyplot as plt
import numpy as np
import os    
import timeit
from ccpi.filters.gpu_regularizers import Diff4thHajiaboli, NML, GPU_ROF_TV
from ccpi.filters.cpu_regularizers_cython import ROF_TV
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################
def rmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(a * b))    
    return rmse
        
filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")

# read image
Im = plt.imread(filename)                     
Im = np.asarray(Im, dtype='float32')

Im = Im/255
perc = 0.075
u0 = Im + np.random.normal(loc = Im ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
# map the u0 u0->u0>0
f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = f(u0).astype('float32')

## plot 
fig = plt.figure(1)
plt.suptitle('Comparison of ROF-TV regularizer using CPU and GPU implementations')
a=fig.add_subplot(1,4,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")


# set parameters
pars = {'algorithm': ROF_TV , \
        'input' : u0,\
        'regularization_parameter':0.04,\
        'time_marching_parameter': 0.0025,\
        'number_of_iterations': 600
        }
print ("#################ROF TV CPU#####################")
start_time = timeit.default_timer()
rof_cpu = ROF_TV(pars['input'],
             pars['number_of_iterations'],
             pars['regularization_parameter'],
             pars['time_marching_parameter'] 
             )
#tgv = out
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


print ("#################ROF TV GPU#####################")
start_time = timeit.default_timer()
rof_gpu = GPU_ROF_TV(pars['input'], 
                     pars['number_of_iterations'], 
                     pars['time_marching_parameter'], 
                     pars['regularization_parameter'])
                     
rms = rmse(Im, rof_gpu)
pars['rmse'] = rms
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
tolerance = 1e-06
diff_im = abs(rof_cpu - rof_gpu)
diff_im[diff_im > tolerance] = 1
a=fig.add_subplot(1,4,4)
imgplot = plt.imshow(diff_im, vmin=0, vmax=1, cmap="gray")
plt.title('{}'.format('Pixels larger threshold difference'))
if (diff_im.sum() > 1):
    print ("Arrays do not match!")
else:
    print ("Arrays match")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:24:26 2018

@author: ofn77899
"""



import matplotlib.pyplot as plt
import numpy as np
import os    
from enum import Enum
import timeit
from ccpi.filters.gpu_regularizers import Diff4thHajiaboli, NML
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
        
filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")
#filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-FISTA_reconstruction\data\lena_gray_512.tif"
#filename = r"/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/data/lena_gray_512.tif"
#filename = r'/home/algol/Documents/Python/STD_test_images/lena_gray_512.tif'

Im = plt.imread(filename)                     
Im = np.asarray(Im, dtype='float32')

perc = 0.15
u0 = Im + np.random.normal(loc = Im ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
# map the u0 u0->u0>0
f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = f(u0).astype('float32')

## plot 
fig = plt.figure()

a=fig.add_subplot(2,3,1)
a.set_title('noise')
imgplot = plt.imshow(u0#,cmap="gray"
                     )


## Diff4thHajiaboli
start_time = timeit.default_timer()
pars = {'algorithm' : Diff4thHajiaboli , \
        'input' : u0,
        'regularization_parameter':0.02 , \
'number_of_iterations' :150 ,\
'edge_preserving_parameter':0.001 
}
d4h = Diff4thHajiaboli(pars['input'], 
                     pars['regularization_parameter'], 
                     pars['number_of_iterations'], 
                     pars['edge_preserving_parameter'])
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(2,3,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(d4h #, cmap="gray"
                     )

a=fig.add_subplot(2,3,5)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, 'd4h - u0', transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(d4h - u0 #, cmap="gray"
                     )


## Patch Based Regul NML
start_time = timeit.default_timer()

pars = {'algorithm' : NML , \
        'input' : u0,
        'SearchW_real':3 , \
'SimilW' :1,\
'h':0.05 ,#
'lambda' : 0.08
}
nml = NML(pars['input'], 
                     pars['SearchW_real'], 
                     pars['SimilW'], 
                     pars['h'],
                     pars['lambda'])
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(2,3,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(nml #, cmap="gray"
                     )

a=fig.add_subplot(2,3,6)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, 'nml - u0', transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(nml - u0 #, cmap="gray"
                     )

plt.show()
        
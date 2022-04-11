#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:38:31 2022

@author: algol
"""
from tomophantom import TomoP3D
import numpy as np
import os
import tomophantom
from tomophantom.supp.artifacts import _Artifacts_
#from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from ccpi.filters.regularisers import PD_TV
#import matplotlib.pyplot as plt


model = 11 # select a model
N_size = 200 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
phantom_3D = TomoP3D.Model(model, N_size, path_library3D)


# forming dictionaries with artifact types
_noise_ =  {'noise_type' : 'Gaussian',
            'noise_sigma' : 0.01, # noise amplitude
            'noise_seed' : 0}


phantom_3D_noisy = _Artifacts_(phantom_3D.copy(), **_noise_)
phantom_3D_noisy /= np.max(phantom_3D_noisy)
phantom_3D_noisy += phantom_3D.copy()

#plt.figure(1)
#plt.rcParams.update({'font.size': 21})
#plt.imshow(phantom_3D_noisy[:,:,100], cmap="BuPu")

#%%
gpu_device = 0

pars = {'algorithm' : PD_TV, \
        'input' : phantom_3D_noisy,\
        'regularisation_parameter':0.1, \
        'number_of_iterations' :1000,\
        'tolerance_constant':0.0,\
        'methodTV': 0 ,\
        'nonneg': 1,
        'lipschitz_const' : 8}

(pd_gpu,info_vec_gpu) = PD_TV(pars['input'],
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'],
              pars['methodTV'],
              pars['nonneg'],
              pars['lipschitz_const'],
              gpu_device)

#plt.figure(2)
#plt.rcParams.update({'font.size': 21})
#plt.imshow(pd_gpu, cmap="BuPu")
#%%

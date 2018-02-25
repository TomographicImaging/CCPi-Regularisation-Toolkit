# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:10:05 2017

@author: ofn77899
"""


import matplotlib.pyplot as plt
import numpy as np
import os    
from enum import Enum
import timeit
from ccpi.filters.cpu_regularizers_boost import SplitBregman_TV , FGP_TV ,\
                                                 LLT_model, PatchBased_Regul ,\
                                                 TGV_PD
from ccpi.filters.cpu_regularizers_cython import ROF_TV

###############################################################################
#https://stackoverflow.com/questions/13875989/comparing-image-in-url-to-image-in-filesystem-in-python/13884956#13884956
#NRMSE a normalization of the root of the mean squared error
#NRMSE is simply 1 - [RMSE / (maxval - minval)]. Where maxval is the maximum
# intensity from the two images being compared, and respectively the same for
# minval. RMSE is given by the square root of MSE: 
# sqrt[(sum(A - B) ** 2) / |A|],
# where |A| means the number of elements in A. By doing this, the maximum value
# given by RMSE is maxval.

def nrmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(a * b))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))
    
def rmse(im1, im2):
    a, b = im1.shape
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(a * b))    
    return rmse
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
#
#  2D Regularizers
#
###############################################################################
#Example:
# figure;
# Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
# u = SplitBregman_TV(single(u0), 10, 30, 1e-04);

# assumes the script is launched from the test directory
filename = os.path.join(".." , ".." , ".." , "data" ,"lena_gray_512.tif")
#filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-FISTA_reconstruction\data\lena_gray_512.tif"
#filename = r"/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/data/lena_gray_512.tif"
#filename = r'/home/algol/Documents/Python/STD_test_images/lena_gray_512.tif'

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
fig = plt.figure()

a=fig.add_subplot(2,4,1)
a.set_title('noise')
imgplot = plt.imshow(u0,cmap="gray"
                     )

reg_output = []
##############################################################################
# Call regularizer

####################### SplitBregman_TV #####################################
# u = SplitBregman_TV(single(u0), 10, 30, 1e-04);

start_time = timeit.default_timer()
pars = {'algorithm' : SplitBregman_TV , \
        'input' : u0,
        'regularization_parameter':15. , \
        'number_of_iterations' :40 ,\
        'tolerance_constant':0.0001 , \
        'TV_penalty': 0
}

out = SplitBregman_TV (pars['input'], pars['regularization_parameter'],
                              pars['number_of_iterations'],
                              pars['tolerance_constant'],
                              pars['TV_penalty'])  
splitbregman = out[0]
rms = rmse(Im, splitbregman)
pars['rmse'] = rms
txtstr = printParametersToString(pars) 
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
    

a=fig.add_subplot(2,4,2)


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(splitbregman,\
                     cmap="gray"
                     )

###################### FGP_TV #########################################
# u = FGP_TV(single(u0), 0.05, 100, 1e-04);
start_time = timeit.default_timer()
pars = {'algorithm' : FGP_TV , \
        'input' : u0,
        'regularization_parameter':0.05, \
        'number_of_iterations' :200 ,\
        'tolerance_constant':1e-4,\
        'TV_penalty': 0
}

out = FGP_TV (pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['TV_penalty'])  

fgp = out[0]
rms = rmse(Im, fgp)
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)       


a=fig.add_subplot(2,4,3)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
imgplot = plt.imshow(fgp, \
                     cmap="gray"
                     )
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

###################### LLT_model #########################################

start_time = timeit.default_timer()

pars = {'algorithm': LLT_model , \
        'input' : u0,
        'regularization_parameter': 5,\
        'time_step':0.00035, \
        'number_of_iterations' :350,\
        'tolerance_constant':0.0001,\
        'restrictive_Z_smoothing': 0
}
out = LLT_model(pars['input'], 
                pars['regularization_parameter'],
                pars['time_step'] , 
                pars['number_of_iterations'],
                pars['tolerance_constant'],
                pars['restrictive_Z_smoothing'] )

llt = out[0]
rms = rmse(Im, out[0])
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(2,4,4)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(llt,\
                     cmap="gray"
                     )


# ###################### PatchBased_Regul #########################################
# # Quick 2D denoising example in Matlab:   
# #   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# #   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
# #   ImDen = PB_Regul_CPU(single(u0), 3, 1, 0.08, 0.05); 

start_time = timeit.default_timer()

pars = {'algorithm': PatchBased_Regul , \
        'input' : u0,
        'regularization_parameter': 0.05,\
        'searching_window_ratio':3, \
        'similarity_window_ratio':1,\
        'PB_filtering_parameter': 0.06
}
out = PatchBased_Regul(pars['input'], 
                       pars['regularization_parameter'],
                       pars['searching_window_ratio'] , 
                       pars['similarity_window_ratio'] , 
                       pars['PB_filtering_parameter'])
pbr = out[0]
rms = rmse(Im, out[0])
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

a=fig.add_subplot(2,4,5)


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(pbr ,cmap="gray")


# ###################### TGV_PD #########################################
# # Quick 2D denoising example in Matlab:   
# #   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# #   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
# #   u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550);

start_time = timeit.default_timer()

pars = {'algorithm': TGV_PD , \
        'input' : u0,\
        'regularization_parameter':0.07,\
        'first_order_term': 1.3,\
        'second_order_term': 1, \
        'number_of_iterations': 550
        }
out = TGV_PD(pars['input'],
             pars['regularization_parameter'],
             pars['first_order_term'] , 
             pars['second_order_term'] , 
             pars['number_of_iterations'])
tgv = out[0]
rms = rmse(Im, out[0])
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(2,4,6)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(tgv, cmap="gray")

# ###################### ROF_TV #########################################

start_time = timeit.default_timer()

pars = {'algorithm': ROF_TV , \
        'input' : u0,\
        'regularization_parameter':0.04,\
        'marching_step': 0.0025,\
        'number_of_iterations': 300
        }
rof = ROF_TV(pars['input'],
             pars['number_of_iterations'],
             pars['regularization_parameter'],
             pars['marching_step'] 
             )
#tgv = out
rms = rmse(Im, rof)
pars['rmse'] = rms

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(2,4,7)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof, cmap="gray")

plt.show()

################################################################################
##
##  3D Regularizers
##
################################################################################
##Example:
## figure;
## Im = double(imread('lena_gray_256.tif'))/255;  % loading image
## u0 = Im + .05*randn(size(Im)); u0(u0 < 0) = 0;
## u = SplitBregman_TV(single(u0), 10, 30, 1e-04);
#
##filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-Reconstruction\python\test\reconstruction_example.mha"
#filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-Simpleflex\data\head.mha"
#
#reader = vtk.vtkMetaImageReader()
#reader.SetFileName(os.path.normpath(filename))
#reader.Update()
##vtk returns 3D images, let's take just the one slice there is as 2D
#Im = Converter.vtk2numpy(reader.GetOutput())
#Im = Im.astype('float32')
##imgplot = plt.imshow(Im)
#perc = 0.05
#u0 = Im + (perc* np.random.normal(size=np.shape(Im)))
## map the u0 u0->u0>0
#f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
#u0 = f(u0).astype('float32')
#converter = Converter.numpy2vtkImporter(u0, reader.GetOutput().GetSpacing(),
#                                        reader.GetOutput().GetOrigin())
#converter.Update()
#writer = vtk.vtkMetaImageWriter()
#writer.SetInputData(converter.GetOutput())
#writer.SetFileName(r"C:\Users\ofn77899\Documents\GitHub\CCPi-FISTA_reconstruction\data\noisy_head.mha")
##writer.Write()
#
#
### plot 
#fig3D = plt.figure()
##a=fig.add_subplot(3,3,1)
##a.set_title('Original')
##imgplot = plt.imshow(Im)
#sliceNo = 32
#
#a=fig3D.add_subplot(2,4,1)
#a.set_title('noise')
#imgplot = plt.imshow(u0.T[sliceNo])
#
#reg_output3d = []
#
###############################################################################
## Call regularizer
#
######################## SplitBregman_TV #####################################
## u = SplitBregman_TV(single(u0), 10, 30, 1e-04);
#
##reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
#
##out = reg(input=u0, regularization_parameter=10., #number_of_iterations=30,
##          #tolerance_constant=1e-4, 
##          TV_Penalty=Regularizer.TotalVariationPenalty.l1)
#
#out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10., number_of_iterations=30,
#          tolerance_constant=1e-4, 
#          TV_Penalty=Regularizer.TotalVariationPenalty.l1)
#
#
#pars = out2[-2]
#reg_output3d.append(out2)
#
#a=fig3D.add_subplot(2,4,2)
#
#
#textstr = out2[-1]
#
#
## these are matplotlib.patch.Patch properties
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
## place a text box in upper left in axes coords
#a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#imgplot = plt.imshow(reg_output3d[-1][0].T[sliceNo])
#
####################### FGP_TV #########################################
## u = FGP_TV(single(u0), 0.05, 100, 1e-04);
#out2 = Regularizer.FGP_TV(input=u0, regularization_parameter=0.005,
#                          number_of_iterations=200)
#pars = out2[-2]
#reg_output3d.append(out2)
#
#a=fig3D.add_subplot(2,4,2)
#
#
#textstr = out2[-1]
#
#
## these are matplotlib.patch.Patch properties
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
## place a text box in upper left in axes coords
#a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#imgplot = plt.imshow(reg_output3d[-1][0].T[sliceNo])
#
####################### LLT_model #########################################
## * u0 = Im + .03*randn(size(Im)); % adding noise
## [Den] = LLT_model(single(u0), 10, 0.1, 1);
##Den = LLT_model(single(u0), 25, 0.0003, 300, 0.0001, 0); 
##input, regularization_parameter , time_step, number_of_iterations,
##                  tolerance_constant, restrictive_Z_smoothing=0
#out2 = Regularizer.LLT_model(input=u0, regularization_parameter=25,
#                          time_step=0.0003,
#                          tolerance_constant=0.0001,
#                          number_of_iterations=300)
#pars = out2[-2]
#reg_output3d.append(out2)
#
#a=fig3D.add_subplot(2,4,2)
#
#
#textstr = out2[-1]
#
#
## these are matplotlib.patch.Patch properties
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
## place a text box in upper left in axes coords
#a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#imgplot = plt.imshow(reg_output3d[-1][0].T[sliceNo])
#
####################### PatchBased_Regul #########################################
## Quick 2D denoising example in Matlab:   
##   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
##   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
##   ImDen = PB_Regul_CPU(single(u0), 3, 1, 0.08, 0.05); 
#
#out2 = Regularizer.PatchBased_Regul(input=u0, regularization_parameter=0.05,
#                          searching_window_ratio=3,
#                          similarity_window_ratio=1,
#                          PB_filtering_parameter=0.08)
#pars = out2[-2]
#reg_output3d.append(out2)
#
#a=fig3D.add_subplot(2,4,2)
#
#
#textstr = out2[-1]
#
#
## these are matplotlib.patch.Patch properties
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
## place a text box in upper left in axes coords
#a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#imgplot = plt.imshow(reg_output3d[-1][0].T[sliceNo])
#

###################### TGV_PD #########################################
# Quick 2D denoising example in Matlab:   
#   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
#   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
#   u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550);


#out2 = Regularizer.TGV_PD(input=u0, regularization_parameter=0.05,
#                          first_order_term=1.3,
#                          second_order_term=1,
#                          number_of_iterations=550)
#pars = out2[-2]
#reg_output3d.append(out2)
#
#a=fig3D.add_subplot(2,4,2)
#
#
#textstr = out2[-1]
#
#
## these are matplotlib.patch.Patch properties
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
## place a text box in upper left in axes coords
#a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
#        verticalalignment='top', bbox=props)
#imgplot = plt.imshow(reg_output3d[-1][0].T[sliceNo])

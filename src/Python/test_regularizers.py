# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:10:05 2017

@author: ofn77899
"""

#from ccpi.viewer.CILViewer2D import Converter
#import vtk

import matplotlib.pyplot as plt
import numpy as np
import os    
from enum import Enum
import timeit
#from PIL import Image
#from Regularizer import Regularizer
from ccpi.imaging.Regularizer import Regularizer

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
###############################################################################

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

#filename = r"C:\Users\ofn77899\Documents\GitHub\CCPi-FISTA_reconstruction\data\lena_gray_512.tif"
filename = r"/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/data/lena_gray_512.tif"
#filename = r'/home/algol/Documents/Python/STD_test_images/lena_gray_512.tif'

#reader = vtk.vtkTIFFReader()
#reader.SetFileName(os.path.normpath(filename))
#reader.Update()
Im = plt.imread(filename)                     
#Im = Image.open('/home/algol/Documents/Python/STD_test_images/lena_gray_512.tif')/255
#img.show()
Im = np.asarray(Im, dtype='float32')




#imgplot = plt.imshow(Im)
perc = 0.05
u0 = Im + (perc* np.random.normal(size=np.shape(Im)))
# map the u0 u0->u0>0
f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = f(u0).astype('float32')

## plot 
fig = plt.figure()
#a=fig.add_subplot(3,3,1)
#a.set_title('Original')
#imgplot = plt.imshow(Im)

a=fig.add_subplot(2,3,1)
a.set_title('noise')
imgplot = plt.imshow(u0)

reg_output = []
##############################################################################
# Call regularizer

####################### SplitBregman_TV #####################################
# u = SplitBregman_TV(single(u0), 10, 30, 1e-04);

use_object = True
if use_object:
    reg = Regularizer(Regularizer.Algorithm.SplitBregman_TV)
    print (reg.pars)
    reg.setParameter(input=u0)    
    reg.setParameter(regularization_parameter=10.)
    # or 
    # reg.setParameter(input=u0, regularization_parameter=10., #number_of_iterations=30,
              #tolerance_constant=1e-4, 
              #TV_Penalty=Regularizer.TotalVariationPenalty.l1)
    plotme = reg() [0]
    pars = reg.pars
    textstr = reg.printParametersToString() 
    
    #out = reg(input=u0, regularization_parameter=10., #number_of_iterations=30,
              #tolerance_constant=1e-4, 
    #          TV_Penalty=Regularizer.TotalVariationPenalty.l1)
    
#out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10., number_of_iterations=30,
#          tolerance_constant=1e-4, 
#          TV_Penalty=Regularizer.TotalVariationPenalty.l1)

else:
    out2 = Regularizer.SplitBregman_TV(input=u0, regularization_parameter=10. )
    pars = out2[2]
    reg_output.append(out2)
    plotme = reg_output[-1][0]
    textstr = out2[-1]

a=fig.add_subplot(2,3,2)


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(plotme,cmap="gray")

###################### FGP_TV #########################################
# u = FGP_TV(single(u0), 0.05, 100, 1e-04);
out2 = Regularizer.FGP_TV(input=u0, regularization_parameter=0.005,
                          number_of_iterations=200)
pars = out2[-2]

reg_output.append(out2)

a=fig.add_subplot(2,3,3)

textstr = out2[-1]

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
out2 = Regularizer.PatchBased_Regul(input=u0, regularization_parameter=0.05,
                           searching_window_ratio=3,
                           similarity_window_ratio=1,
                           PB_filtering_parameter=0.08)
pars = out2[-2]
reg_output.append(out2)

a=fig.add_subplot(2,3,5)


textstr = out2[-1]

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0])
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0],cmap="gray")

###################### LLT_model #########################################
# * u0 = Im + .03*randn(size(Im)); % adding noise
# [Den] = LLT_model(single(u0), 10, 0.1, 1);
#Den = LLT_model(single(u0), 25, 0.0003, 300, 0.0001, 0); 
#input, regularization_parameter , time_step, number_of_iterations,
#                  tolerance_constant, restrictive_Z_smoothing=0
out2 = Regularizer.LLT_model(input=u0, regularization_parameter=25,
                          time_step=0.0003,
                          tolerance_constant=0.0001,
                          number_of_iterations=300)
pars = out2[-2]

reg_output.append(out2)

a=fig.add_subplot(2,3,4)
out2 = Regularizer.PatchBased_Regul(input=u0, regularization_parameter=0.05,
                           searching_window_ratio=3,
                           similarity_window_ratio=1,
                           PB_filtering_parameter=0.08)
pars = out2[-2]
reg_output.append(out2)

a=fig.add_subplot(2,3,5)


textstr = out2[-1]

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0],cmap="gray")

textstr = out2[-1]
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0],cmap="gray")

# ###################### PatchBased_Regul #########################################
# # Quick 2D denoising example in Matlab:   
# #   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# #   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
# #   ImDen = PB_Regul_CPU(single(u0), 3, 1, 0.08, 0.05); 

out2 = Regularizer.PatchBased_Regul(input=u0, regularization_parameter=0.05,
                       searching_window_ratio=3,
                       similarity_window_ratio=1,
                       PB_filtering_parameter=0.08)
pars = out2[-2]
reg_output.append(out2)

a=fig.add_subplot(2,3,5)


textstr = out2[-1]

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0],cmap="gray")


# ###################### TGV_PD #########################################
# # Quick 2D denoising example in Matlab:   
# #   Im = double(imread('lena_gray_256.tif'))/255;  % loading image
# #   u0 = Im + .03*randn(size(Im)); u0(u0<0) = 0; % adding noise
# #   u = PrimalDual_TGV(single(u0), 0.02, 1.3, 1, 550);


out2 = Regularizer.TGV_PD(input=u0, regularization_parameter=0.05,
                           first_order_term=1.3,
                           second_order_term=1,
                           number_of_iterations=550)
pars = out2[-2]
reg_output.append(out2)

a=fig.add_subplot(2,3,6)


textstr = out2[-1]


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
a.text(0.05, 0.95, textstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(reg_output[-1][0],cmap="gray")


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
#a=fig3D.add_subplot(2,3,1)
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
#a=fig3D.add_subplot(2,3,2)
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
#a=fig3D.add_subplot(2,3,2)
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
#a=fig3D.add_subplot(2,3,2)
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
#a=fig3D.add_subplot(2,3,2)
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
#a=fig3D.add_subplot(2,3,2)
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
